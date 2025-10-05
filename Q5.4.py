import pickle
import os
import matplotlib.pyplot as plt

# Load scheduled learning rate validation accuracy curve
with open("scheduled_lr_val_acc.pkl", "rb") as f:
    sched_val_curve = pickle.load(f)
sched_steps, sched_accs = zip(*sched_val_curve)

# Load Adam run result from part3_cnn4.pkl
with open("part3_cnn4.pkl", "rb") as f:
    adam_dict = pickle.load(f)

# Fix keys if needed
if 'val acc curve' in adam_dict:
    adam_dict['val_acc_curve'] = adam_dict.pop('val acc curve')
if 'test acc' in adam_dict:
    adam_dict['test_acc'] = adam_dict.pop('test acc')

# Extract Adam results
adam_accs = adam_dict['val_acc_curve']
adam_steps = list(range(0, len(adam_accs)*10, 10))  # assuming every 10 steps

# Test accuracies
test_acc_sgd = 0.6571  # from printed result or replace with actual
test_acc_adam = adam_dict['test_acc']

# Plotting
plt.figure(figsize=(14, 6))

# Left: Validation accuracy curves
plt.subplot(1, 2, 1)
plt.plot(sched_steps, sched_accs, label="cnn4_SGD (Scheduled LR)", color='blue')
plt.plot(adam_steps, adam_accs, label="cnn4_Adam", color='red')
plt.xlabel("Step")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)

# Right: Test accuracy comparison
plt.subplot(1, 2, 2)
plt.scatter(["cnn4_SGD", "cnn4_Adam"], [test_acc_sgd, test_acc_adam], color=['blue', 'red'])
plt.ylim(0.65, 0.675)
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy")
plt.grid(True)

# Show and save
plt.tight_layout()
plt.savefig("optimizer_comparison.png")
plt.show()
