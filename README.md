# Interpreting Model Vulnerabilities Through Circuits: A Reproducibility Study in Fairness 
based on ["Detecting and ​Understanding Vulnerabilities​ in Language Models via MI"](https://arxiv.org/pdf/2407.19842)

## Requirements

```setup
pip install -r requirements.txt
```


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

Directories `configs`, `scripts`, `src` are mounted in the Docker image, and should contain all code for readers to reproduce our results: main program contained in `src`, experiment settings contained in `configs`, and `scripts` that run experiments using `config` and `src`.

Local directories `cache`, `data`, `work` are for managing your work. These are in `gitignore` and are not commited. 

`snellius_env` is for us to keep track of our job files.

### Example workflow

#### Local development workflow

Since `src`, `scripts`, and `configs` directories are mounted in real-time from your local machine, changes to your code appear immediately inside the containear.

  1. Modify code or script.
  2. Ensure Docker Desktop is set to Linux container mode on Windows (or simply use Docker on macOS).
  3. Pull the latest image from Docker Hub: `docker pull lesterpjy10/refact-multiarch:latest`
  4. Set test/run alias (Remove `--gpus all` for local without gpu) 
     ```
     alias dtest="docker run --rm -it \
	 -v $(pwd)/data:/local/data \
	 -v $(pwd)/cache:/local/cache \
	 -v $(pwd)/work:/local/work \
	 -v $(pwd)/src:/local/src \
	 -v $(pwd)/scripts:/local/scripts \
	 -v $(pwd)/configs:/local/configs \
	 --gpus all"
  5. Run an interactive test: `dtest lesterpjy10/refact-multiarch:latest bash`
  6. Or, for environment sanity check, run `dtest lesterpjy10/refact-multiarch python scripts/test_env.py` 
 
#### Committing and tagging for release

  1. Once you are satisfied with local testing, **add-commit-push** to your branch (this does not build the image)
  2. To release an image:
     ```
     git tag build-* 
     git push origin build-*
     ```
     replace `*` with an unique name. Alternatively, builds could be triggered in the Github action UI for the default branch.
  3. Github Action will spin up, build linux/amd64 + linux/arm64 images, and push them to Docker Hub as:
     ```
     lesterpjy10/refact-multiarch:latest
     lesterpjy10/refact-multiarch:build-*
     ```

#### Test and run image on Snellius

  1. Pull your changes on Snellius
  2. Edit user name in job file `snellius_env/dockerim2sif.job` Replace `/tmp/scur2818XXXX` for `APPTAINER_TMPDIR` with your own user name, if your SURF username is `user01`, change to `/tmp/user01XXXX`
  3. Run `sbatch snellius_env/dockerim2sif.job` to pull Docker image from Dockerhub, and convert image to sif file for Apptainer.
  4. Check sif file successfully built by inspecting output file `work/build_container_*.out`
  5. Run `sbatch snellius_env/test_env.job` to test container environment with `Apptainer run`
  6. Check package successfully installed by inspecting the output file under `work` directory.
  7. `test_env.py` output should look like:
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

