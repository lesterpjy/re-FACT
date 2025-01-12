#!/usr/bin/env python3

import sys
import torch
import torchvision
import torchaudio
import pytorch_lightning as pl
import tensorboard
import tabulate
import tqdm
from PIL import Image
import notebook
import jupyterlab
import matplotlib
import seaborn
import ipywidgets

# Try to get version from importlib.metadata
try:
    import importlib.metadata
    tqdm_version = importlib.metadata.version("tqdm")
except (ImportError, importlib.metadata.PackageNotFoundError):
    tqdm_version = "unknown"

def main():
    print("---- Package Versions ----")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"PyTorch Lightning version: {pl.__version__}")
    print(f"TensorBoard version: {tensorboard.__version__}")
    print(f"Tabulate version: {tabulate.__version__}")
    print(f"TQDM version: {tqdm_version}")
    print(f"Pillow (PIL) version: {Image.__version__}")
    print(f"Notebook version: {notebook.__version__}")
    print(f"JupyterLab version: {jupyterlab.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"Seaborn version: {seaborn.__version__}")
    print(f"ipywidgets version: {ipywidgets.__version__}")
    print("---- End of Versions ----\n")

    # Check CUDA availability
    gpu_available = torch.cuda.is_available()
    print(f"CUDA GPU Available: {gpu_available}")
    if gpu_available:
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Running on CPU only.")

    # Simple forward pass on a random tensor
    print("\n---- Running a tiny sanity-check forward pass with PyTorch ----")
    model = torch.nn.Linear(10, 5)  # just a small linear model
    data = torch.randn(2, 10)       # batch of size 2, 10 features
    output = model(data)
    print("Input shape:", data.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

    print("\nEnvironment functional.")

if __name__ == "__main__":
    main()

