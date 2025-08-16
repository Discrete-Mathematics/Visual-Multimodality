import os
from PIL import Image


def crop_images(input_folder, output_folder, target_resolution=(2048, 1536), crop_box=(462, 311, 1544, 1243)):
    """
    Batch crop images, processing only those matching the target resolution

    Parameters:
    input_folder: Path to input image directory
    output_folder: Path to output image directory
    target_resolution: Required resolution (width, height)
    crop_box: Crop region (left, upper, right, lower)
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    # Counters
    processed_count = 0
    skipped_count = 0

    print(f"Starting image processing...")
    print(f"Source directory: {input_folder}")
    print(f"Target directory: {output_folder}")
    print(f"Required resolution: {target_resolution[0]}x{target_resolution[1]}")
    print(f"Crop region: {crop_box}\n")

    # Process all files
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Open image and verify resolution
                with Image.open(input_path) as img:
                    print(f"Processing: {filename} | Dimensions: {img.size[0]}x{img.size[1]}")

                    # Validate resolution
                    if img.size != target_resolution:
                        print(
                            f"  - Skipped: Incorrect resolution (expected {target_resolution[0]}x{target_resolution[1]})")
                        skipped_count += 1
                        continue

                    # Perform cropping
                    cropped_img = img.crop(crop_box)

                    # Save result (preserve original format)
                    cropped_img.save(output_path)
                    print(f"  - Cropped successfully! Saved to: {output_path}")
                    processed_count += 1

            except Exception as e:
                print(f"  - Error processing {filename}: {str(e)}")
                skipped_count += 1

    # Processing summary
    print("\nProcessing completed!")
    print(f"Total images processed: {processed_count + skipped_count}")
    print(f"  - Successfully cropped: {processed_count}")
    print(f"  - Skipped: {skipped_count}")
    if processed_count > 0:
        print(f"Cropped images saved to: {output_folder}")


if __name__ == "__main__":
    # Configuration parameters
    INPUT_DIR = "Path/To/Folder"  # Input image directory
    OUTPUT_DIR = "Path/To/Folder"  # Output directory

    # Crop region (left, upper, right, lower)
    CROP_BOX = (462, 311, 1544, 1243)

    # Target resolution (width, height)
    TARGET_RESOLUTION = (2048, 1536)

    # Execute cropping
    crop_images(INPUT_DIR, OUTPUT_DIR, TARGET_RESOLUTION, CROP_BOX)
