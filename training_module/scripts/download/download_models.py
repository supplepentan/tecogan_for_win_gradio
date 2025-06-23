import argparse
import hashlib
import os
import sys

import requests


def download_gdrive_file(file_id, dest_path):
    """
    Downloads a file from Google Drive, handling large file warnings.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    print(f"  Downloading file with ID: {file_id}")
    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest_path)


def get_confirm_token(response):
    """
    Retrieves the confirmation token from the cookies of a Google Drive response.
    """
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, dest_path):
    """
    Saves the content of a requests.Response object to a file.
    """
    CHUNK_SIZE = 32768
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def check_md5(filepath, expected_md5):
    """
    Verifies the MD5 checksum of a file.
    """
    print(f"  Verifying MD5 checksum for {os.path.basename(filepath)}...")
    if not os.path.exists(filepath):
        print(f"  Error: File not found at {filepath}")
        return False

    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)

    if md5.hexdigest() == expected_md5:
        print("  MD5 checksum verified.")
        return True
    else:
        print(f"!!! MD5 checksum mismatch for: {filepath}")
        print(f"!!! Expected {expected_md5}, but got {md5.hexdigest()}")
        print("!!! Please try downloading it again.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pretrained TecoGAN models.")
    parser.add_argument(
        "degradation",
        type=str,
        choices=["BD", "BI"],
        help="Degradation type ('BD' or 'BI').",
    )
    parser.add_argument(
        "model_name",
        type=str,
        choices=["TecoGAN", "FRVSR"],
        help="Model name ('TecoGAN' or 'FRVSR').",
    )
    args = parser.parse_args()

    # List of models to download
    # Format: (Degradation, Model Name, Filename, GDrive File ID, MD5 Checksum)
    MODELS = [
        (
            "BD",
            "TecoGAN",
            "TecoGAN_BD_iter500000.pth",
            "13FPxKE6q7tuRrfhTE7GB040jBeURBj58",
            "13d826c9f066538aea9340e8d3387289",
        ),
        (
            "BD",
            "FRVSR",
            "FRVSR_BD_iter400000.pth",
            "11kPVS04a3B3k0SD-mKEpY_Q8WL7KrTIA",
            "77d33c58b5cbf1fc68a1887be80ed18f",
        ),
        (
            "BI",
            "TecoGAN",
            "TecoGAN_BI_iter500000.pth",
            "1ie1F7wJcO4mhNWK8nPX7F0LgOoPzCwEu",
            "4955b65b80f88456e94443d9d042d1e6",
        ),
        (
            "BI",
            "FRVSR",
            "FRVSR_BI_iter400000.pth",
            "1wejMAFwIBde_7sz-H7zwlOCbCvjt3G9L",
            "ad6337d934ec7ca72441082acd80c4ae",
        ),
    ]

    # Find the model to download
    target_model = next(
        (m for m in MODELS if m[0] == args.degradation and m[1] == args.model_name),
        None,
    )

    if not target_model:
        print(
            f"Error: No model found for combination {args.degradation} and {args.model_name}"
        )
        sys.exit(1)

    _, _, filename, file_id, expected_md5 = target_model

    output_dir = "./pretrained_models"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f">>> Model [{filename}] already exists. Skipping.")
        return

    print(f">>> Start to download model [{args.degradation} {args.model_name}]")
    os.makedirs(output_dir, exist_ok=True)

    download_gdrive_file(file_id, output_path)

    if not check_md5(output_path, expected_md5):
        sys.exit(1)

    print(f"  Successfully downloaded model to {output_path}")


if __name__ == "__main__":
    main()
