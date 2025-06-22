import argparse
import os
import subprocess
import sys


def main():
    """
    Python script to evaluate a pretrained model, replacing test.sh.
    """
    parser = argparse.ArgumentParser(description="Evaluate a pretrained TecoGAN model.")
    parser.add_argument(
        "degradation",
        type=str,
        help="Degradation type, e.g., 'BD' or 'BI'.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Model path relative to experiments folder, e.g., 'TecoGAN/TecoGAN_REDS_4xSR_2GPU'.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="GPU IDs to use, comma-separated (e.g., '0' or '0,1').",
    )
    parser.add_argument(
        "--master_port",
        type=str,
        default="4322",
        help="Master port for distributed launch (used for multi-GPU).",
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

    gpu_list = args.gpu_ids.split(",")
    num_gpus = len(gpu_list)

    if num_gpus > 1:
        cmd.extend(
            [
                "-m",
                "torch.distributed.launch",
                f"--nproc_per_node={num_gpus}",
                f"--master_port={args.master_port}",
            ]
        )

    cmd.extend(
        [
            os.path.join(root_dir, "codes", "main.py"),
            "--exp_dir",
            exp_dir,
            "--mode",
            "test",
            "--opt",
            args.opt,
            "--gpu_ids",
            args.gpu_ids,
        ]
    )

    # --- Set environment and run the command ---
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    print(f"Executing command:\n{' '.join(cmd)}\n")
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
