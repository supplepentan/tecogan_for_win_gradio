from typing import Any, Dict, Union
from torch.utils.data import DataLoader
from .datasets_utils import ImageFolderDataset


def create_dataloader(
    opt: Dict[str, Any], phase: str, idx: Union[str, int]
) -> DataLoader:
    """
    データローダーを作成する関数。

    Parameters:
        :param opt: 設定オプションを含む辞書。
        :param phase: 実行フェーズ（例: "test"）。
        :param idx: 使用するデータセットのインデックス。

    Returns:
        :return: PyTorchのDataLoaderオブジェクト。
    """
    # データセットに関する設定を取得
    data_opt = opt["dataset"].get(idx)

    if phase == "test":
        # テストフェーズ用のデータローダーを作成
        loader = DataLoader(
            dataset=ImageFolderDataset(
                data_opt
            ),  # データセットをImageFolderDatasetでラップ
            batch_size=1,  # バッチサイズを1に設定（テスト時は通常1シーケンスごとに処理）
            shuffle=False,  # テスト時はシャッフルしない
            num_workers=data_opt[
                "num_worker_per_gpu"
            ],  # データの読み込みに使用するスレッド数
            pin_memory=data_opt["pin_memory"],  # CUDAピンメモリを使用するかどうか
        )
    else:
        # "test" フェーズ以外の場合はエラーを投げる
        raise ValueError(f"Unrecognized phase: {phase}")

    return loader  # 作成したデータローダーを返す
