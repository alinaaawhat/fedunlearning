# fedunlearning

Minimal **CIFAR-100** FedAvg checkpoint + `fft_fedavg_unlearn.py` (spectral continual unlearning, ResNet-18).

## Setup

```bash
cd submit_unlearning
pip install -r requirements.txt
export PYTHONPATH=.
```

Put `fedavg_cifar100_alpha0.01_le10.pth` under `results/` or pass `--ckpt_pth`.

## Run

```bash
python fft_fedavg_unlearn.py
```

Dataset cache: `dataset/cifar100/`.
