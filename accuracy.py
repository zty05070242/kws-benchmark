"""Train and evaluate the four keyword spotters on Speech Commands v2."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import tarfile
import wave

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchaudio

from benchmark import BCResNets, tc_resnet8
from extensions import ds_cnn_small, matchboxnet


WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
CLASSES = (*WORDS, "_unknown_", "_silence_")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BACKGROUND_CATEGORIES = {
    "rain", "sea_waves", "crackling_fire", "crickets", "wind", "pouring_water",
    "washing_machine", "vacuum_cleaner",
}


def unpack(stem, marker):
    target = DATA / stem
    complete = target / ".complete"
    if not complete.exists():
        target.mkdir(exist_ok=True)
        with tarfile.open(DATA / f"{stem}.tar.gz") as archive:
            archive.extractall(target, filter="data")
        if not (target / marker).exists():
            raise ValueError(f"Unexpected archive layout: {target}")
        complete.touch()
    return target


def read_wave(path):
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError(f"Expected 16 kHz mono PCM16: {path}")
        samples = np.frombuffer(wav.readframes(16000), dtype="<i2")
    clip = torch.zeros(16000)
    clip[:len(samples)] = torch.from_numpy(samples.astype(np.float32)).div_(32768)
    return clip


def background_clips(split, count):
    """Draw one-second excerpts from source-disjoint ESC-50 folds."""
    root = DATA / "esc50"
    with (root / "esc50.csv").open() as metadata:
        rows = [row for row in csv.DictReader(metadata)
                if row["category"] in BACKGROUND_CATEGORIES]
    folds = {"train": {1, 2, 3}, "validation": {4, 5}}
    sources = {
        name: {row["src_file"] for row in rows if int(row["fold"]) in group}
        for name, group in folds.items()
    }
    assert sources["train"].isdisjoint(sources["validation"])
    resample = torchaudio.transforms.Resample(44100, 16000)
    recordings = []
    for row in rows:
        if int(row["fold"]) not in folds[split]:
            continue
        with wave.open(str(root / "audio" / row["filename"]), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (44100, 1, 2)
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        recordings.append(resample(torch.from_numpy(pcm.astype(np.float32)).div_(32768)))

    generator = torch.Generator().manual_seed(1001 if split == "train" else 1002)
    clips = []
    for _ in range(count):
        recording = recordings[torch.randint(len(recordings), (), generator=generator).item()]
        start = torch.randint(recording.numel() - 15999, (), generator=generator).item()
        clip = recording[start:start + 16000]
        # Vary level from -65 to -15 dBFS RMS, with a floor for quiet excerpts.
        rms = 10 ** ((-65 + 50 * torch.rand((), generator=generator).item()) / 20)
        clips.append((clip * (rms / clip.square().mean().sqrt().clamp_min(1e-4))).clamp(-1, 1))
    return clips


def split_files():
    corpus = unpack("speech_commands_v0.02", "validation_list.txt")
    test = unpack("speech_commands_test_set_v0.02", "yes")
    validation = set((corpus / "validation_list.txt").read_text().splitlines())
    held_out = set((corpus / "testing_list.txt").read_text().splitlines())
    assert validation.isdisjoint(held_out)
    splits = {"train": [], "validation": []}
    for path in sorted(corpus.glob("*/*.wav")):
        relative = path.relative_to(corpus).as_posix()
        if path.parent.name.startswith("_") or relative in held_out:
            continue
        splits["validation" if relative in validation else "train"].append(path)

    def label(path):
        word = path.parent.name
        return CLASSES.index(word) if word in CLASSES else 10

    result = {}
    for seed, (split, files) in enumerate(splits.items(), 101):
        targets = [(path, label(path)) for path in files if label(path) < 10]
        unknown = [path for path in files if label(path) == 10]
        per_class = round(len(targets) / len(WORDS))
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(unknown), generator=generator)[:per_class]
        result[split] = targets + [(unknown[i], 10) for i in order]
        result[split] += [(None, 11)] * per_class
    result["test"] = [(path, label(path)) for path in sorted(test.glob("*/*.wav"))]
    assert len(result["test"]) == 4890
    return result


def frontend(name):
    if name in ("tc", "ds", "matchbox"):
        window, hop, n_fft, n_mels, coefficients = {
            "tc": (480, 160, 480, 40, 40),
            "ds": (640, 320, 640, 40, 10),
            "matchbox": (400, 160, 512, 64, 64),
        }[name]
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=coefficients, log_mels=name != "matchbox",
            melkwargs={"n_fft": n_fft, "win_length": window, "hop_length": hop,
                       "n_mels": n_mels, "center": name == "matchbox"},
        )
        if name == "ds":
            return lambda waveform: mfcc(waveform).transpose(1, 2).unsqueeze(1).contiguous()
        if name == "matchbox":
            return lambda waveform: nn.functional.pad(mfcc(waveform), (13, 14))
        return mfcc
    # Same front end as the released BC-ResNet implementation.
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=512, win_length=480, hop_length=160, n_mels=40
    )
    return lambda waveform: (mel(waveform) + 1e-6).log().unsqueeze(1)


def features(name, split, rows, background="synthetic"):
    variant = "-esc50" if background == "esc50" and split != "test" else ""
    cache = DATA / f"{name}-{split}{variant}-features.pt"
    if cache.exists():
        return TensorDataset(*torch.load(cache, weights_only=True))
    transform = frontend(name)
    generator = torch.Generator().manual_seed(1001 if split == "train" else 1002)
    real_clips = iter(background_clips(split, sum(path is None for path, _ in rows))) if variant else None
    batches = []
    for start in range(0, len(rows), 128):
        clips = []
        for path, _ in rows[start:start + 128]:
            if path is None and real_clips is not None:
                clip = next(real_clips)
            elif path is None:
                # Synthetic low-level silence avoids borrowing test background recordings.
                clip = torch.randn(16000, generator=generator)
                clip *= torch.rand((), generator=generator) * 0.003
            else:
                clip = read_wave(path)
            clips.append(clip)
        batches.append(transform(torch.stack(clips)))
    data = (torch.cat(batches), torch.tensor([label for _, label in rows]))
    torch.save(data, cache.with_suffix(".tmp"))
    cache.with_suffix(".tmp").replace(cache)
    print(name, split, len(rows), dict(sorted(Counter(data[1].tolist()).items())), flush=True)
    return TensorDataset(*data)


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    predictions = []
    for x, y in data:
        predictions.append(model(x.to(device)).argmax(1).cpu())
    return torch.cat(predictions)


def run(name, args, splits):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = {"tc": tc_resnet8, "bc": lambda: BCResNets(8),
             "ds": ds_cnn_small, "matchbox": matchboxnet}[name]().to(device)
    torch.manual_seed(args.seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    if args.smoke:
        x = frontend(name)(torch.randn(4, 16000)).to(device)
        assert x.shape[1:] == {
            "tc": (40, 98), "bc": (1, 40, 101), "ds": (1, 49, 10), "matchbox": (64, 128),
        }[name]
        optimiser.zero_grad()
        loss = nn.functional.cross_entropy(model(x), torch.arange(4, device=device))
        loss.backward()
        optimiser.step()
        assert torch.isfinite(loss)
        print(name, x.shape, f"one training step, loss={loss.item():.4f}", flush=True)
        return
    datasets = {split: features(name, split, splits[split], args.background)
                for split in ("train", "validation")}
    train = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed))
    validation = DataLoader(datasets["validation"], batch_size=args.batch_size)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    best = -1
    suffix = "-esc50" if args.background == "esc50" else ""
    checkpoint = ROOT / f"{name}-best-{args.seed}{suffix}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train:
            optimiser.zero_grad()
            nn.functional.cross_entropy(model(x.to(device)), y.to(device)).backward()
            optimiser.step()
        schedule.step()
        accuracy = (predict(model, validation, device) == datasets["validation"].tensors[1]).float().mean().item()
        print(f"{name} epoch={epoch} validation_accuracy={accuracy:.6f}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save(model.state_dict(), checkpoint.with_suffix(".tmp"))
            checkpoint.with_suffix(".tmp").replace(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test = features(name, "test", splits["test"])
    targets = test.tensors[1]
    predictions = predict(model, DataLoader(test, batch_size=args.batch_size), device)
    correct = predictions == targets
    print(f"{name} held_out_test_accuracy={correct.float().mean().item():.6f}", flush=True)
    for index, label in enumerate(CLASSES):
        mask = targets == index
        print(f"{name} {label}: {correct[mask].sum().item()}/{mask.sum().item()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tc", "bc", "ds", "matchbox", "both", "extensions", "all"),
                        default="both")
    parser.add_argument("--background", choices=("synthetic", "esc50"), default="synthetic")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(f"torch={torch.__version__} {args}", flush=True)
    torch.set_num_threads(args.threads)
    splits = None if args.smoke else split_files()
    models = {"both": ("tc", "bc"), "extensions": ("ds", "matchbox"),
              "all": ("tc", "bc", "ds", "matchbox")}
    for name in models.get(args.model, (args.model,)):
        run(name, args, splits)


if __name__ == "__main__":
    main()
