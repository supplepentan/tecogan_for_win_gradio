import argparse
import os
import subprocess
import sys


def main():
    """
    Python script to profile a TecoGAN model, replacing profile.sh.
    """
    parser = argparse.ArgumentParser(
        description="Profile a TecoGAN model (FLOPs, parameters, speed)."
    )
    parser.add_argument(
        "degradation",
        type=str,
        help="Degradation type, e.g., 'BD' or 'BI'.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model path relative to experiments folder, e.g., 'TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU'.",
    )
    parser.add_argument(
        "lr_size",
        type=str,
        help="Size of the LR video in format [channels]x[height]x[width], e.g., '3x134x320'.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="GPU IDs to use, comma-separated (e.g., '0' or '0,1').",
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="test.yml",
        help="Name of the configuration file to use.",
    )
    args = parser.parse_args()

    root_dir = "."
    exp_dir = os.path.join(root_dir, f"experiments_{args.degradation}", args.model)

    # --- Construct the command ---
    cmd = [sys.executable]  # Use the same python interpreter that runs this script
    cmd.extend(
        [
            os.path.join(root_dir, "codes", "main.py"),
            "--exp_dir",
            exp_dir,
            "--mode",
            "profile",
            "--opt",
            args.opt,
            "--gpu_ids",
            args.gpu_ids,
            "--lr_size",
            args.lr_size,
            "--test_speed",  # Always test speed when profiling via this script
        ]
    )

    print(f"Executing command:\n{' '.join(cmd)}\n")
    subprocess.run(cmd)  # No need for env=env as gpu_ids is passed as arg


if __name__ == "__main__":
    main()
