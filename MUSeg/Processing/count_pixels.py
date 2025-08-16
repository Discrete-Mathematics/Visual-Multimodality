import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# Define input and output folders
input_folder = 'Path/To/Folder'
output_folder = 'Path/To/Folde'

# Create output folder
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Use defaultdict to automatically create new brightness groups
brightness_groups = defaultdict(list)

# Process all files in input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        parts = filename.split('-')
        if len(parts) >= 2:
            brightness = parts[-2]  # Get second-to-last group as brightness value

            # Read image
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                # Calculate valid pixel ratio
                non_zero_pixels = np.count_nonzero(img)
                total_pixels = img.size
                ratio = non_zero_pixels / total_pixels
                percent_ratio = round(ratio * 100, 2)

                # Store results (auto-create new brightness groups)
                brightness_groups[brightness].append((filename, percent_ratio))

# Sort brightness groups naturally
sorted_brightness = sorted(brightness_groups.keys(), key=lambda x: int(x) if x.isdigit() else x)

# Process each brightness group
for brightness in sorted_brightness:
    data_list = brightness_groups[brightness]

    # Create group result file
    output_file = os.path.join(output_folder, f"{brightness}.txt")

    # Initialize bin counts for statistics
    bin_counts = [0] * 10  # 10 bins: 0-10%, 10-20%, ..., 90-100%

    with open(output_file, 'w') as f:
        # Write results for each image
        for filename, ratio in data_list:
            f.write(f"{filename}, {ratio:.2f}%\n")

            # Count bin distribution (handle edge cases)
            bin_index = min(max(0, int(ratio // 10)), 9)
            bin_counts[bin_index] += 1

        # Add statistics section
        f.write("\n[Statistics]\n")
        total_images = len(data_list)
        for i in range(10):
            lower_bound = i * 10
            upper_bound = (i + 1) * 10
            count = bin_counts[i]
            percentage = (count / total_images) * 100 if total_images > 0 else 0

            # Handle last bin specially
            if i == 9:
                range_str = "90-100%"
            else:
                range_str = f"{lower_bound}-{upper_bound}%"

            f.write(f"{range_str}: {percentage:.1f}% ({count} images)\n")

print(f"Processing complete! Generated {len(brightness_groups)} brightness group files")
