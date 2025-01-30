import io 
import pickle
import torch
import os

class CPU_Unpickler(pickle.Unpickler):
    """
    For loading pickled files on CPU when they were saved on GPU.
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_pickle_from_gpu(file_path):
    """
    Load a pickled file that was saved on GPU.
    """
    with open(file_path, 'rb') as f:
        return CPU_Unpickler(f).load()
    

def load_pickle(file_path):
    """
    Load a pickled file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    

def load_files_named(output_folder, file_starts_with='mask', if_gpu=True):

    files = [
        file for file in os.listdir(output_folder)
        if file.startswith(file_starts_with) and file.endswith('.pkl')
    ]

    # Sort the filenames
    files.sort()

    loaded_files = []

    # Iterate through the sorted files and load tensors
    for file in files:

        file_path = os.path.join(output_folder, file)

        with open(file_path, 'rb') as f:

            if if_gpu:

                loaded_files.append(load_pickle_from_gpu(file_path))
            
            else:

                loaded_files.append(load_pickle(file_path))

    return loaded_files

def filter_adv_samples(adv_samples, pad_token = 128009):

    # Find the maximum token length
    max_length = max(sample.size(1) for sample in adv_samples)

    # Pad all tensors to the maximum length using the padding token 128009
    padded_samples = []
    for sample in adv_samples:
        pad_length = max_length - sample.size(1)
        if pad_length > 0:
            # Add padding tokens
            pad_tensor = torch.full((sample.size(0), pad_length), pad_token, dtype=sample.dtype, device=sample.device)
            sample = torch.cat((sample, pad_tensor), dim=1)
        padded_samples.append(sample)

    # Concatenate all padded tensors
    final_tensor = torch.cat(padded_samples, dim=0)

    return torch.unique(final_tensor, sorted=False, dim=0)


def filter_adv_org_samples(adv_samples, org_samples, pad_token=128009):
    """
    Filters and pads adversarial samples and retrieves corresponding original samples.
    
    Parameters:
        adv_samples (list of torch.Tensor): List of adversarial samples.
        org_samples (list of torch.Tensor): List of original samples corresponding to adv_samples.
        pad_token (int): Token used for padding.
    
    Returns:
        torch.Tensor: Unique padded adversarial samples.
        torch.Tensor: Corresponding original samples aligned with the unique adversarial samples.
    """
    # Ensure org_samples and adv_samples have the same length
    assert len(adv_samples) == len(org_samples), "adv_samples and org_samples must have the same length."
    
    # Find the maximum token length
    max_length = max(sample.size(1) for sample in adv_samples)
    
    # Pad all tensors to the maximum length using the padding token
    padded_adv_samples = []
    padded_org_samples = []
    for adv_sample, org_sample in zip(adv_samples, org_samples):
        pad_length = max_length - adv_sample.size(1)
        
        if pad_length > 0:
            # Add padding tokens to adversarial samples
            pad_tensor_adv = torch.full((adv_sample.size(0), pad_length), pad_token, 
                                     dtype=adv_sample.dtype, device=adv_sample.device)
            adv_sample = torch.cat((adv_sample, pad_tensor_adv), dim=1)
            
            # Add padding tokens to original samples
            pad_tensor_org = torch.full((org_sample.size(0), pad_length), pad_token,
                                      dtype=org_sample.dtype, device=org_sample.device)
            org_sample = torch.cat((org_sample, pad_tensor_org), dim=1)
        
        padded_adv_samples.append(adv_sample)
        padded_org_samples.append(org_sample)
    
    # Concatenate all padded tensors
    final_adv_tensor = torch.cat(padded_adv_samples, dim=0)
    final_org_tensor = torch.cat(padded_org_samples, dim=0)
    
    # Get unique adversarial samples and their indices
    unique_adv_tensor, inverse_indices, counts = torch.unique(
        final_adv_tensor, 
        sorted=False, 
        dim=0, 
        return_inverse=True,
        return_counts=True
    )
    
    # Create a mapping from unique indices to first occurrence in original data
    unique_to_original = {}
    for idx, inv_idx in enumerate(inverse_indices):
        if inv_idx.item() not in unique_to_original:
            unique_to_original[inv_idx.item()] = idx
    
    # Get the indices of the first occurrence of each unique element
    first_occurrence_indices = torch.tensor(
        [unique_to_original[i] for i in range(len(unique_adv_tensor))],
        device=final_org_tensor.device
    )
    
    # Use these indices to get the corresponding original samples
    corresponding_org_tensor = final_org_tensor[first_occurrence_indices]
    
    return unique_adv_tensor, corresponding_org_tensor
