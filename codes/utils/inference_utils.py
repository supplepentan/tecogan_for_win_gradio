# inference_utils.py
from pathlib import Path
from typing import Dict, Any

from codes.models import define_model
from codes.utils import (
    dist_utils,
    base_utils,
    data_utils,
    dataloader_utils,
)


def inference(opt: Dict[str, Any]) -> None:
    """
    モデルを使用して推論を実行し、パフォーマンスを評価する関数。

    Parameters:
        :param opt: 設定オプションを含む辞書。
    """
    # モデルごとに推論と評価を実行
    for load_path in opt["model"]["generator"]["load_path_lst"]:
        # モデルインデックスを設定（ファイル名から拡張子を除いた部分を使用）
        model_idx = Path(load_path).stem

        # モデルを作成
        opt["model"]["generator"]["load_path"] = load_path
        model = define_model(opt)

        # 各テストデータセットに対して処理を行う
        for dataset_idx in sorted(opt["dataset"].keys()):
            # テストデータセット以外はスキップ
            if "test" not in dataset_idx:
                continue

            ds_name = opt["dataset"][dataset_idx]["name"]
            print("ds_name:", ds_name)
            base_utils.log_info(
                f"Testing on {ds_name} dataset"
            )  # テストデータセット名をログに記録

            # データローダーを作成
            test_loader = dataloader_utils.create_dataloader(
                opt, phase="test", idx=dataset_idx
            )
            test_dataset = test_loader.dataset
            num_seq = len(test_dataset)  # データセット内のシーケンス数を取得

            # メトリクス計算ツールを作成（コメントアウトされている）
            # metric_calculator = create_metric_calculator(opt)

            # 各シーケンスに対して推論を実行
            rank, world_size = dist_utils.get_dist_info()  # 分散情報を取得
            for idx in range(rank, num_seq, world_size):
                # データを取得
                data = test_dataset[idx]

                # データを推論用に準備
                model.prepare_inference_data(data)

                # 推論を実行
                hr_seq = model.infer()

                # 推論結果を保存
                if opt["test"]["save_res"]:
                    # Revert to using opt["test"]["res_dir"] as the base,
                    # ensuring it's correctly set by TecoGanProcessor via base_utils.opt
                    base_output_dir = Path(opt["test"]["res_dir"])
                    res_dir = (
                        base_output_dir / ds_name
                    )  # ds_name should be "resolved_images"
                    # res_seq_dir was defined but not used for save_sequence,
                    # which takes res_dir directly.
                    # print("res_dir for saving:", str(res_dir))
                    # print("data['seq_idx'] (used for subfolder if res_seq_dir was used):", data["seq_idx"])

                    # Ensure hr_seq is not None and contains data before saving
                    # (Add checks for hr_seq content here if not done already)
                    data_utils.save_sequence(
                        str(res_dir), hr_seq, data["frm_idx"], to_bgr=True
                    )
            base_utils.log_info("-" * 40)  # 区切り線をログに出力
