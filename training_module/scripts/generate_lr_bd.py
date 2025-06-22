import argparse
import os
import cv2
import glob
from tqdm import tqdm


def generate_lr_images(hr_dir, lr_dir, scale, sigma):
    """
    Generates low-resolution (LR) images from high-resolution (HR) images
    using Gaussian blur followed by downsampling (BD degradation).

    Args:
        hr_dir (str): Directory containing the high-resolution source images.
        lr_dir (str): Directory where the low-resolution images will be saved.
        scale (int): The downscaling factor (e.g., 4 for 4x SR).
        sigma (float): The sigma value for the Gaussian blur.
    """
    # Find all image files recursively in the HR directory
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "**", "*.png"), recursive=True))

    if not hr_files:
        print(f"Error: No image files found in '{hr_dir}'")
        return

    print(f"Found {len(hr_files)} HR images. Starting LR image generation...")

    # Calculate kernel size for Gaussian blur from sigma. It must be an odd number.
    ksize = int(sigma * 3.0) * 2 + 1

    for hr_path in tqdm(hr_files, desc="Generating LR images"):
        # Create the corresponding LR path, preserving the subdirectory structure
        relative_path = os.path.relpath(hr_path, hr_dir)
        lr_path = os.path.join(lr_dir, relative_path)

        # Create the output subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(lr_path), exist_ok=True)

        hr_img = cv2.imread(hr_path)
        if hr_img is None:
            print(f"Warning: Could not read image '{hr_path}'. Skipping.")
            continue

        blurred_img = cv2.GaussianBlur(hr_img, (ksize, ksize), sigma)

        h, w, _ = blurred_img.shape
        # Ensure the dimensions are divisible by the scale factor before resizing
        h_lr, w_lr = h // scale, w // scale
        if h_lr == 0 or w_lr == 0:
            print(
                f"Warning: Image '{hr_path}' is too small to be downscaled by {scale}. Skipping."
            )
            continue

        lr_img = cv2.resize(blurred_img, (w_lr, h_lr), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(lr_path, lr_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LR images with BD degradation."
    )
    parser.add_argument(
        "--hr_dir", type=str, required=True, help="Directory of HR images."
    )
    parser.add_argument(
        "--lr_dir", type=str, required=True, help="Directory to save LR images."
    )
    parser.add_argument("--scale", type=int, default=4, help="Downscaling factor.")
    parser.add_argument(
        "--sigma", type=float, default=1.5, help="Sigma for Gaussian blur."
    )
    args = parser.parse_args()

    generate_lr_images(args.hr_dir, args.lr_dir, args.scale, args.sigma)
    print("\nLR image generation complete.")
