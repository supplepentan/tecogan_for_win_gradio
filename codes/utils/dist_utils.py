import functools
from typing import Callable, Tuple, Any, Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(opt: Dict[str, Any], local_rank: int) -> None:
    """分散トレーニングの初期化を行う関数。

    Parameters:
        :param opt: 設定オプションを含む辞書。
        :param local_rank: ローカルプロセスのランク（GPU番号）。
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank, world_size = get_dist_info()

    opt.update(
        {
            "dist": True,
            "device": "cuda",
            "local_rank": local_rank,
            "world_size": world_size,
            "rank": rank,
        }
    )


def get_dist_info() -> Tuple[int, int]:
    """プロセスのランクとワールドサイズを取得する関数。

    Returns:
        :return: ランク (rank) とワールドサイズ (world_size) のタプル。
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def master_only(func: Callable[..., Any]) -> Callable[..., Any]:
    """マスターのみで関数を実行するデコレータ。

    Parameters:
        :param func: 実行する関数。

    Returns:
        :return: ランクが0のときのみ実行される関数のラッパー。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
