# My Paper Title

This repository is the official implementation of [My Paper Title]


Notebooks location and usage:

### Experiment 1

### Experiment 2

- src/activation_patching/activation_patching_script.py --df_path src/activation_patching/final_toxicity_prompts_0.csv --batch_size 16 --dir_name src/activation_patching/final_toxicity_data_0 --wiki_names_path src/dataset_generation/wiki_last_name_master.csv --calculate_patches --image_output_path src/activation_patching/activation_patching.png  - runs activation patching using authors' method

### Experiment 3

-src/org_paper/grad_experiment_acro.ipynb - notebook used to generate adversarial samples for authors' task
-src/adv_sample/sample_generation.ipynb - notebook used to generate adversarial samples for our task

### Experiment 4

-src/adv_attack/vul_heads_detection.ipynb - notebook for creating results of vulnerable heads using authors' method 

### Experiment 5 

-src/dataset_generation/circuit_adv_data.ipynb - notebook for creating dataset for bias circuit from adversarial samples

-src/bias_scale_results.ipynb - notebook about results of Experiment 5, scaling down the edges to mitigate the bias

### Others

-src/plots.ipynb - notebook for additional plots for the paper
