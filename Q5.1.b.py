import pickle
import os
from utils import part5Plots

# Load the recorded results from the pickle file
with open("part5_cnn4.pkl", "rb") as f:
    results_dict = pickle.load(f)

# Convert keys from "loss curve 1" to "loss_curve_1" and similarly for validation curves.
def convert_keys(results):
    converted = {}
    for key, value in results.items():
        if key.startswith("loss curve"):
            new_key = key.replace(" ", "_")
        elif key.startswith("val acc curve"):
            new_key = key.replace(" ", "_")
        else:
            new_key = key
        converted[new_key] = value
    return converted

results_dict = convert_keys(results_dict)

# Create a directory to save plots if it doesn't exist
save_dir = './plots'
os.makedirs(save_dir, exist_ok=True)

# Generate performance comparison plots using the updated dictionary
part5Plots(results_dict, save_dir=save_dir, filename='part5_cnn4_performance', show_plot=True)
