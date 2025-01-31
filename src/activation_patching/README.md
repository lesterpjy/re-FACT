## Activation Patching

Directory for storing files required for running activation patching.


- activation_patching_script.py - script to run activation patching on toxicity dataset (with corruption of the whole sentence)
- final_toxicity_prompts_0.csv - toxicity dataset (can be passed as an argument to activation_patching_script.py)
- final_toxicity_data_0 - directory with results from activation_patching_script.py (having the results allow for skipping calculating the activation patches, useful for reproduction of the results)
