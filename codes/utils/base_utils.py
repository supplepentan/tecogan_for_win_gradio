# base_utils.py
from pathlib import Path
import random
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from .dist_utils import init_dist, master_only


def opt(output_directory, resolved_images_directory, model) -> Dict[str, Any]:
    opt: Dict[str, Any] = {
        "mode": "test",
        "gpu_ids": "0",
        "local_rank": 0,
        "test_speed": False,
        "is_train": False,
        "scale": 4,
        "manual_seed": 0,
        "verbose": False,
        "dataset": {
            "degradation": {"type": "BD", "sigma": 1.5},
            "test": {
                "name": str(Path(resolved_images_directory).name),
                "lr_seq_dir": "data",
                "num_worker_per_gpu": 3,
                "pin_memory": True,
            },
        },
        "model": {
            "name": "TecoGAN",
            "generator": {
                "name": "FRNet",
                "in_nc": 3,
                "out_nc": 3,
                "nf": 64,
                "nb": 10,
                "load_path": model,
            },
        },
        "test": {
            "save_res": True,
            "res_dir": output_directory,
            "padding_mode": "reflect",
            "num_pad_front": 5,
        },
    }
    setup_device(opt)
    setup_random_seed(opt.get("manual_seed", 2021) + opt["rank"])
    setup_logger("base")
    setup_paths(opt)
    return opt


def setup_device(opt: Dict[str, Any]) -> None:
    opt["gpu_ids"] = tuple(map(int, opt["gpu_ids"].split(",")))
    if opt["gpu_ids"][0] < 0 or not torch.cuda.is_available():
        opt.update({"dist": False, "device": "cpu", "rank": 0})
    else:
        if len(opt["gpu_ids"]) == 1:
            torch.cuda.set_device(0)
            opt.update({"dist": False, "device": "cuda", "rank": 0})
        else:
            init_dist(opt, opt["local_rank"])
        torch.backends.cudnn.benchmark = True


def setup_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name: str) -> None:
    base_logger = logging.getLogger(name=name)
    base_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s]: %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    base_logger.addHandler(sh)


@master_only
def log_info(msg: str, logger_name: str = "base") -> None:
    logger = logging.getLogger(logger_name)
    logger.info(msg)


def print_options(
    opt: Dict[str, Any], logger_name: str = "base", tab: str = ""
) -> None:
    for key, val in opt.items():
        if isinstance(val, dict):
            log_info(f"{tab}{key}:", logger_name)
            print_options(val, logger_name, tab + "  ")
        else:
            log_info(f"{tab}{key}: {val}", logger_name)


def retrieve_files(dir: str, suffix: str = "png|jpg") -> List[Path]:
    """Retrieve files with specific suffix under dir and sub-dirs recursively."""

    def retrieve_files_recursively(dir: Path, file_lst: List[Path]) -> None:
        for d in sorted(dir.iterdir()):
            if d.is_dir():
                retrieve_files_recursively(d, file_lst)
            elif d.suffix.lower() in [f".{s}" for s in suffix.split("|")]:
                file_lst.append(d)

    if not dir:
        return []

    file_lst: List[Path] = []
    retrieve_files_recursively(Path(dir), file_lst)
    file_lst.sort()

    return file_lst


def setup_paths(opt: Dict[str, Any]) -> None:
    def setup_ckpt_dir() -> None:
        ckpt_dir: Path = Path(opt["train"].get("ckpt_dir", ""))
        if not ckpt_dir:
            ckpt_dir = Path(opt["exp_dir"]) / "train" / "ckpt"
            opt["train"]["ckpt_dir"] = str(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    def setup_res_dir() -> None:
        res_dir: Path = Path(opt["test"].get("res_dir", ""))
        if not res_dir:
            res_dir = Path(opt["exp_dir"]) / "test" / "results"
            opt["test"]["res_dir"] = str(res_dir)
        res_dir.mkdir(parents=True, exist_ok=True)

    def setup_json_path() -> None:
        json_dir: Path = Path(opt["test"].get("json_dir", ""))
        if not json_dir:
            json_dir = Path(opt["exp_dir"]) / "test" / "metrics"
            opt["test"]["json_dir"] = str(json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)

    def setup_model_path() -> None:
        load_path: Path = Path(opt["model"]["generator"].get("load_path", ""))
        if not load_path:
            raise ValueError("Pretrained generator model is needed for testing")

        if load_path.stem == "*":
            start_iter: int = opt["test"]["start_iter"]
            end_iter: int = opt["test"]["end_iter"]
            freq: int = opt["test"]["test_freq"]
            opt["model"]["generator"]["load_path_lst"] = [
                str(load_path.parent / f"G_iter{iter}.pth")
                for iter in range(start_iter, end_iter + 1, freq)
            ]
        else:
            opt["model"]["generator"]["load_path_lst"] = [str(load_path)]

    if opt["mode"] == "train":
        setup_ckpt_dir()
        for dataset_idx in opt["dataset"].keys():
            if "test" not in dataset_idx:
                continue
            if opt["test"].get("save_res", False):
                setup_res_dir()
            if opt["test"].get("save_json", False):
                setup_json_path()
    elif opt["mode"] == "test":
        setup_model_path()
        for dataset_idx in opt["dataset"].keys():
            if "test" not in dataset_idx:
                continue
            if opt["test"].get("save_res", False):
                setup_res_dir()
            if opt["test"].get("save_json", False):
                setup_json_path()
