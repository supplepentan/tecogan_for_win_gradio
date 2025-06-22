import argparse
import hashlib
import os
import sys
import time
import zipfile

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
    Saves the content of a requests.Response object to a file with a progress bar.
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
    parser = argparse.ArgumentParser(
        description="Download datasets for TecoGAN model evaluation."
    )
    parser.add_argument(
        "degradation",
        type=str,
        choices=["BD", "BI"],
        help="Degradation type ('BD' or 'BI').",
    )
    args = parser.parse_args()

    # List of datasets to download
    # Format: (Name, Target Directory, Zip Filename, GDrive File ID, MD5 Checksum, Type)
    DATASETS = [
        (
            "Vid4 GT",
            "./data/Vid4/GT",
            "GT",
            "1T8TuyyOxEUfXzCanH5kvNH2iA8nI06Wj",
            "d2850eccf30092418f15afe4a7ea27e5",
            "GT",
        ),
        (
            "ToS3 GT",
            "./data/ToS3/GT",
            "GT",
            "1XoR_NVBR-LbZOA8fXh7d4oPV0M8fRi8a",
            "56eb9e8298a4e955d618c1658dfc89c9",
            "GT",
        ),
        (
            "Vid4 LR (BD)",
            "./data/Vid4/Gaussian4xLR",
            "Gaussian4xLR",
            "1-5NFW6fEPUczmRqKHtBVyhn2Wge6j3ma",
            "3b525cb0f10286743c76950d9949a255",
            "BD",
        ),
        (
            "ToS3 LR (BD)",
            "./data/ToS3/Gaussian4xLR",
            "Gaussian4xLR",
            "1rDCe61kR-OykLyCo2Ornd2YgPnul2ffM",
            "803609a12453a267eb9c78b68e073e81",
            "BD",
        ),
        (
            "Vid4 LR (BI)",
            "./data/Vid4/Bicubic4xLR",
            "Bicubic4xLR",
            "1Kg0VBgk1r9I1c4f5ZVZ4sbfqtVRYub91",
            "35666bd16ce582ae74fa935b3732ae1a",
            "BI",
        ),
        (
            "ToS3 LR (BI)",
            "./data/ToS3/Bicubic4xLR",
            "Bicubic4xLR",
            "1FNuC0jajEjH9ycqDkH4cZQ3_eUqjxzzf",
            "3b165ffc8819d695500cf565bf3a9ca2",
            "BI",
        ),
    ]

    # Filter datasets based on user input
    datasets_to_download = [
        d for d in DATASETS if d[5] == "GT" or d[5] == args.degradation
    ]

    for name, target_dir, zip_filename, file_id, md5, _ in datasets_to_download:
        if os.path.exists(target_dir):
            print(f">>> Dataset [{name}] already exists. Skipping.")
            continue

        print(f">>> Start to download [{name}] dataset")

        # Define paths
        data_dir = os.path.dirname(target_dir)
        zip_path = os.path.join(data_dir, f"{zip_filename}.zip")

        # Create directory
        os.makedirs(data_dir, exist_ok=True)

        # Download
        download_gdrive_file(file_id, zip_path)

        # Verify
        if not check_md5(zip_path, md5):
            sys.exit(1)

        # Unzip and clean up
        print(f"  Unzipping {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        print(f"  Successfully downloaded and extracted {name}.")

        # Pause between downloads
        time.sleep(1)

    print("\nAll required datasets have been downloaded.")


if __name__ == "__main__":
    main()
