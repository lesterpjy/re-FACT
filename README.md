>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 


## Docker reproducibility management

Directories `configs`, `scripts`, `src` are in the Docker image, and should contain all code for readers to reproduce our results: main program contained in `src`, experiment settings contained in `configs`, and `scripts` that run experiments using `config` and `src`.

Locally there should be `cache`, `data`, `work` directories for you to manage your work. These are in `gitignore` and we should never commit them. 

`snellius_env` is for us to keep track of our job files.

#### To build the Multi-arch Docker Image for amd64/arm64

Enable Buildx:
`docker buildx create --name mybuilder --use`

Confirm Buildx is active:
`docker buildx ls`

Build Multi-Arch image and push to DockerHub (this might take a while, we should add CD/CI to automate this)
```
docker buildx build --platform linux/amd64,linux/arm64 -t lesterpjy10/refact-base-image:latest --push .
```

### Test environment locally

Create alias
```
alias drun="docker run --rm -it -v $(pwd)/data:/local/data \
			 -v $(pwd)/cache:/local/cache -v $(pwd)/work:/local/work --gpus all"
alias dtest="docker run --rm -it -v $(pwd)/data:/local/data \
                         -v $(pwd)/cache:/local/cache \
                         -v $(pwd)/work:/local/work \
                         -v $(pwd)/src:/local/src \
                         -v $(pwd)/scripts:/local/scripts \
                         -v $(pwd)/configs:/local/configs \
                         --gpus all"
```
Remove `--gpus all` for local without gpu

Test environment:
`dtest lesterpjy10/refact-base-image python scripts/test_env.py`

### Test environment on Snellius

Edit user name in job file `snellius_env/dockerim2sif.job`
Replace `/tmp/scur2818XXXX` for `APPTAINER_TMPDIR` with your own user name, if your SURF username is `user01`, change to `/tmp/user01XXXX`

- Run `sbatch snellius_env/dockerim2sif.job` to pull Docker image from Dockerhub, and convert image to sif file for Apptainer.
- Check sif file successfully built by inspecting output file `work/build_container_*.out`
- Run `sbatch snellius_env/test_env.job` to test container environment with `Apptainer run`
- Check package successfully installed by inspecting the output file under `work` directory.

Sample `test_env.py` output

```
---- Package Versions ----
Python version: 3.12.8 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:31:09) [GCC 11.2.0]
PyTorch version: 2.5.0+cu118
Torchvision version: 0.20.0+cu118
Torchaudio version: 2.5.0+cu118
PyTorch Lightning version: 2.4.0
TensorBoard version: 2.17.1
Tabulate version: 0.9.0
TQDM version: 4.66.5
Pillow (PIL) version: 11.0.0
Notebook version: 7.3.2
JupyterLab version: 4.3.4
Matplotlib version: 3.10.0
Seaborn version: 0.13.2
ipywidgets version: 8.1.5
---- End of Versions ----

CUDA GPU Available: True
Using GPU: NVIDIA A100-SXM4-40GB

---- Running a tiny sanity-check forward pass with PyTorch ----
Input shape: torch.Size([2, 10])
Output shape: torch.Size([2, 5])
Output: tensor([[-0.0179, -0.1882,  0.0772, -0.1044, -0.3526],
        [-0.5355,  0.9435, -0.0136, -0.2977,  0.4146]],
       grad_fn=<AddmmBackward0>)

Environment functional.

JOB STATISTICS
==============
Job ID: 9204342
Cluster: snellius
User/Group: —-
State: COMPLETED (exit code 0)
Nodes: 1
Cores per node: 18
CPU Utilized: 00:00:10
CPU Efficiency: 2.42% of 00:06:54 core-walltime
Job Wall-clock time: 00:00:23
Memory Utilized: 2.41 MB
Memory Efficiency: 0.01% of 31.25 GB
```

