#!/usr/bin/env python3

# torch==2.5.0
# torchvision==0.20.0
# torchaudio==2.5.0

# pytorch-lightning==2.4.0
# tabulate>=0.9.0
# tqdm>=4.66.5
# pillow>=10.4.0
# notebook>=7.2.2
# jupyterlab>=4.2.5
# matplotlib>=3.9.2
# seaborn>=0.13.2
# ipywidgets>=8.1.2
# plotly==5.17.0
# einops==0.8.0
# jaxtyping==0.2.28
# numpy==2.2.1
# pandas==2.2.3
# transformer-lens @ git+https://github.com/neelnanda-io/TransformerLens/
# loguru==0.7.3
# transformers==4.48.0ls
# tokenizers==0.21.0
# python-dotenv==1.0.1
# pygraphviz==1.14
# huggingface-hub==0.27.1
# einops==0.8.0
# datasets==3.2.0
# cmapy==0.6.6

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
import plotly
import einops
import jaxtyping
import numpy
import pandas
import transformer_lens
import loguru
import transformers
import tokenizers
import dotenv
import pygraphviz
import huggingface_hub
import datasets
import cmapy


# Try to get version from importlib.metadata
try:
    import importlib.metadata

    tqdm_version = importlib.metadata.version("tqdm")
except (ImportError, importlib.metadata.PackageNotFoundError):
    tqdm_version = "unknown"


def main():
    print(f"Einops version: {einops.__version__}")
    print(f"Jaxtyping version: {jaxtyping.__version__}")
    print(f"Numpy version: {numpy.__version__}")
    print(f"Pandas version: {pandas.__version__}")
    print(f"Transformer Lens version: {transformer_lens.__version__}")
    print(f"Loguru version: {loguru.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"Tokenizers version: {tokenizers.__version__}")
    print(f"Python-dotenv version: {dotenv.__version__}")
    print(f"Pygraphviz version: {pygraphviz.__version__}")
    print(f"Huggingface Hub version: {huggingface_hub.__version__}")
    print(f"Datasets version: {datasets.__version__}")
    print(f"Cmapy version: {cmapy.__version__}")
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
    data = torch.randn(2, 10)  # batch of size 2, 10 features
    output = model(data)
    print("Input shape:", data.shape)
    print("Output shape:", output.shape)
    print("Output:", output)

    print("\nEnvironment functional.")


if __name__ == "__main__":
    main()
