import pickle
import os
from utils import part3Plots, visualizeWeights

# Create a list of model names you have saved
model_names = ["mlp1", "mlp2", "cnn3", "cnn4", "cnn5"]

# This will store the loaded-and-fixed result dictionaries
results_fixed = []

for model_name in model_names:
    filename = f"part3_{model_name}.pkl"
    if not os.path.isfile(filename):
        print(f"File {filename} not found, skipping.")
        continue

    with open(filename, "rb") as f:
        result_dict = pickle.load(f)

    # Fix the key names so that part3Plots does not complain:
    # (Change only if they exist in the dictionary)
    if 'loss curve' in result_dict:
        result_dict['loss_curve'] = result_dict.pop('loss curve')
    if 'train acc curve' in result_dict:
        result_dict['train_acc_curve'] = result_dict.pop('train acc curve')
    if 'val acc curve' in result_dict:
        result_dict['val_acc_curve'] = result_dict.pop('val acc curve')
    if 'test acc' in result_dict:
        result_dict['test_acc'] = result_dict.pop('test acc')

    # Append to our fixed-results list
    results_fixed.append(result_dict)

# Now call part3Plots on the fixed dictionaries
save_dir = './plots'
part3Plots(results_fixed, save_dir=save_dir, filename='part3_performance_comparison', show_plot=True)

# Finally, visualize weights for each architecture
for result in results_fixed:
    model_name = result['name']
    weights = result['weights']
    visualizeWeights(weights, save_dir=save_dir, filename=f'weights_{model_name}')
