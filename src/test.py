# Example of calling the function
from config_utils import load_config
import os

config = load_config("configs/llama3_config.py")


def list_directory_contents(directory):
    try:
        return os.listdir(directory)
    except FileNotFoundError:
        return f"Directory {directory} not found."


data_dir_contents = list_directory_contents(config.data_dir)
work_dir_contents = list_directory_contents(config.work_dir)

print("Contents of data_dir:", data_dir_contents)
print("Contents of work_dir:", work_dir_contents)
