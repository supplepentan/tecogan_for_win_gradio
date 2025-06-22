import argparse
import os
import sys
import shutil
import subprocess


def main():
    """
    Python script to launch TecoGAN model training.
    This script replaces the functionality of train.sh for a more portable and
    flexible training setup, especially on Windows.
    """
    parser = argparse.ArgumentParser(
        description="Python script to train a TecoGAN model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("degradation", type=str, help="Degradation type (e.g., BD, BI)")
    parser.add_argument(
        "model",
        type=str,
        help="Model name path (e.g., TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="GPU IDs to use, comma-separated (e.g., 0,1). Set to -1 for CPU.",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=4321,
        help="Master port for distributed training.",
    )
    parser.add_argument(
        "--start_iter",
        type=int,
        default=0,
        help="Starting iteration for resuming training.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (backs up the `codes` directory).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing experiment directory if it exists.",
    )

    args = parser.parse_args()

    # The script is expected to be run from the `training_module` directory.
    root_dir = "."

    # --- Suffix for resuming training ---
    suffix = f"_iter{args.start_iter}" if args.start_iter > 0 else ""

    # --- Experiment directory setup ---
    exp_dir = os.path.join(root_dir, f"experiments_{args.degradation}", args.model)
    train_dir = os.path.join(exp_dir, "train")

    if os.path.exists(train_dir):
        if args.force:
            print(
                f"Warning: --force flag is set. Removing existing directory: {train_dir}"
            )
            shutil.rmtree(
                exp_dir
            )  # Remove the parent directory to ensure a clean start
        else:
            print(
                f">> Experiment directory already exists: {train_dir}", file=sys.stderr
            )
            print(
                ">> Please delete it, use a different model name, or use the --force flag to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

    os.makedirs(train_dir, exist_ok=True)
    print(f"Created experiment directory: {train_dir}")

    # --- Backup codes if in debug mode ---
    if args.debug:
        codes_dir = os.path.join(root_dir, "codes")
        backup_dir = os.path.join(train_dir, f"codes_backup{suffix}")
        if os.path.exists(codes_dir):
            print(f"Backing up 'codes' directory to {backup_dir}")
            shutil.copytree(codes_dir, backup_dir)
        else:
            print(
                f"Warning: 'codes' directory not found at {codes_dir}. Cannot perform backup.",
                file=sys.stderr,
            )

    # --- Prepare the training command ---
    if args.gpu_ids == "-1":
        num_gpus = 0
    else:
        num_gpus = len(args.gpu_ids.split(","))

    # Use the same Python interpreter that is running this script
    command = [sys.executable]

    if num_gpus > 1:
        command.extend(
            [
                "-m",
                "torch.distributed.launch",
                "--nproc_per_node",
                str(num_gpus),
                "--master_port",
                str(args.master_port),
            ]
        )

    main_script_path = os.path.join(root_dir, "codes", "main.py")
    command.extend(
        [
            main_script_path,
            "--exp_dir",
            exp_dir,
            "--mode",
            "train",
            "--opt",
            f"train{suffix}.yml",
            "--gpu_ids",
            args.gpu_ids,
        ]
    )

    # --- Run the command ---
    env = os.environ.copy()
    if num_gpus > 0:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    log_file_path = os.path.join(train_dir, f"train{suffix}.log")

    print("\n" + "=" * 80)
    print("Starting TecoGAN Training")
    print(f"  - Command: {' '.join(command)}")
    print(f"  - Log file: {log_file_path}")
    print("Training will run in the background. Check the log file for progress.")
    print("=" * 80 + "\n")

    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            command, env=env, stdout=log_file, stderr=subprocess.STDOUT
        )

    print(f"Process launched successfully with PID: {process.pid}")


if __name__ == "__main__":
    main()
