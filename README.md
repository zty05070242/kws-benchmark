# Small-footprint keyword spotting benchmark

Code and recorded results for the four-model comparison in *Accuracy–Efficiency Trade-offs in CNN-Based Small-Footprint Keyword Spotting: A Review*. The benchmark compares TC-ResNet8, BC-ResNet-1, DS-CNN-S and MatchboxNet on the same 12-class Speech Commands v2 task. It reports clip accuracy, weight storage, counted MACs in convolutional and linear layers, and CPU inference latency after feature extraction.

## Setup

Run these commands from the repository root. The reported runs used Python 3.12 and the package versions in `requirements.txt`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python accuracy.py --model all --smoke
```

The smoke check runs one training step for each model with synthetic input. The full experiment below uses the speech recordings and the official test archive.

## Datasets

Download both [Google Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands) archives and the pinned [ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6) recordings. The scripts unpack Speech Commands and cache features under `data/`. Allow at least 12 GB of free space for the downloads, audio and features.

```sh
mkdir -p data
curl -fL -o data/speech_commands_v0.02.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
curl -fL -o data/speech_commands_test_set_v0.02.tar.gz https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
curl -fL -o data/esc50.zip https://github.com/karolpiczak/ESC-50/archive/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6.zip
unzip -q data/esc50.zip -d data
mv data/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6 data/esc50
cp data/esc50/meta/esc50.csv data/esc50/esc50.csv
```

## Reproduce the comparison

Training used Apple Metal (`mps`) on a Mac. Use `--device cpu` on a machine without MPS. Accuracy can vary across software and hardware, and the latency values are specific to the timing platform.

```sh
mkdir -p runs
for seed in 42 43 44; do
  python accuracy.py --model all --background esc50 --seed "$seed" --epochs 15 --batch-size 64 --threads 4 --device mps > "runs/accuracy-esc50-seed${seed}.log" || exit 1
done
python benchmark.py --extensions --esc50-seed 42 > runs/benchmark.json
```

Each accuracy log ends each model's run with its test accuracy and per-class counts. The benchmark loads the four seed-42 checkpoints and the first official test clip's precomputed features. It reports the median of eight block-average inference times for each model. The training logs and timing JSON underlying the paper are in [`results/`](results).

## Model sources

- TC-ResNet8 is a PyTorch port of the [Hyperconnect TensorFlow classifier](https://github.com/hyperconnect/TC-ResNet/tree/8ccbff3a45590247d8c54cc82129acb90eecf5c8). The port implements the classifier head and omits the auxiliary range head.
- BC-ResNet-1 uses the unchanged [Qualcomm AI Research model files](https://github.com/Qualcomm-AI-research/bcresnet/tree/656f947a2a757ab5f36f649ed303ea0191cf553e) under [their original licence](bcresnet/LICENSE).
- DS-CNN-S follows the architecture and small-model configuration in [Arm's Hello Edge repository](https://github.com/ARM-software/ML-KWS-for-MCU).
- MatchboxNet follows NVIDIA's NGC `commandrecognition_en_matchboxnet3x2x64_v2_subset_task` configuration (version `1.0.0rc1`) in a PyTorch implementation.

The TC-ResNet and Arm repositories use the [Apache 2.0 licence](licenses/Apache-2.0.txt). Codex assisted with preparing the scripts.
