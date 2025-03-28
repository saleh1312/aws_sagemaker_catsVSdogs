import json
import os
from dotenv import load_dotenv


load_dotenv()

# Read directories from environment variables (with default values)
hyper_params_dir = os.getenv("HYPERPARAMS_PATH")

# Ensure the directories exist
if not os.path.exists(hyper_params_dir):
    raise Exception(f"hyper params directory not found: {hyper_params_dir}")


with open(hyper_params_dir, "r") as file:
    hyperparams = json.load(file)

print(hyperparams)

# Access specific hyperparameters
learning_rate = float(hyperparams["learning_rate"])
batch_size = int(hyperparams["batch_size"])
num_epochs = int(hyperparams["num_epochs"])

print(f"Learning Rate: {learning_rate}, Batch Size: {batch_size}, Epochs: {num_epochs}")
