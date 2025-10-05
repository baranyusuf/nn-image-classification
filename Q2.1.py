import numpy as np
import matplotlib.pyplot as plt
from utils import part2Plots

def my_conv2d(input_data, kernel):

    # Unpack shapes
    batch_size, in_channels, in_height, in_width = input_data.shape
    out_channels, kernel_in_channels, filter_height, filter_width = kernel.shape

    # Sanity check
    assert in_channels == kernel_in_channels, (
        f"Input has {in_channels} channels, but kernel expects {kernel_in_channels}"
    )

    # Compute output spatial dimensions (no padding, stride=1)
    out_height = in_height - filter_height + 1
    out_width = in_width - filter_width + 1

    # Initialize the output array
    out = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)

    # Perform the convolution
    for b in range(batch_size):
        for oc in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    # "Slice" the input region
                    region = input_data[b, :, i:i + filter_height, j:j + filter_width]
                    # Element-wise multiply with kernel and sum
                    out[b, oc, i, j] = np.sum(region * kernel[oc, :, :, :])

    return out

# 1. Load your data
input_data = np.load(r"C:\EE449\data\samples_7.npy")
kernel = np.load(r"C:\EE449\data\kernel.npy")

# 2. Run convolution
out = my_conv2d(input_data, kernel)

# 3. Save output if needed
np.save("out.npy", out)

# 4. Visualize output
part2Plots(out)
plt.show()