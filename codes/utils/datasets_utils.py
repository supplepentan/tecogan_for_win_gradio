from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def retrieve_files(dir: Union[str, Path], suffix: str = "png|jpg") -> List[Path]:
    """指定されたディレクトリとサブディレクトリ内の特定の拡張子（pngやjpg）のファイルを再帰的に取得する関数"""

    def retrieve_files_recursively(dir: Path, file_lst: List[Path]) -> None:
        """指定ディレクトリ内を再帰的に探索してファイルリストを取得"""
        for d in sorted(dir.iterdir()):
            if d.is_dir():
                # ディレクトリの場合、再帰的に探索
                retrieve_files_recursively(d, file_lst)
            elif d.suffix.lower() in [f".{s}" for s in suffix.split("|")]:
                # 指定された拡張子に一致するファイルをリストに追加
                file_lst.append(d)

    dir = Path(dir)
    if not dir.exists():
        return []

    file_lst: List[Path] = []
    retrieve_files_recursively(dir, file_lst)
    file_lst.sort()

    return file_lst


class BaseDataset(Dataset):
    def __init__(self, data_opt: dict, **kwargs: Optional[dict]) -> None:
        """基本的なデータセットクラス。設定オプションを受け取り、クラス属性として設定"""
        # data_optの内容をクラス属性として設定
        for kw, args in data_opt.items():
            setattr(self, kw, args)

        # kwargsで指定されたオプションで上書き可能
        for kw, args in kwargs.items():
            setattr(self, kw, args)

    def __len__(self) -> int:
        """データセットのサイズを返す（実装はサブクラスで行う）"""
        raise NotImplementedError

    def __getitem__(self, item: int) -> torch.Tensor:
        """指定されたインデックスのデータを返す（実装はサブクラスで行う）"""
        raise NotImplementedError

    def check_info(self, gt_keys: List[str], lr_keys: List[str]) -> None:
        """GT（高解像度）とLR（低解像度）の画像リストが一致しているか確認する関数"""
        if len(gt_keys) != len(lr_keys):
            raise ValueError(
                f"GT & LR contain different numbers of images ({len(gt_keys)}  vs. {len(lr_keys)})"
            )

        for i, (gt_key, lr_key) in enumerate(zip(gt_keys, lr_keys)):
            gt_info = self.parse_lmdb_key(gt_key)
            lr_info = self.parse_lmdb_key(lr_key)

            if gt_info[0] != lr_info[0]:
                raise ValueError(
                    f"video index mismatch ({gt_info[0]} vs. {lr_info[0]} for the {i} key)"
                )

            gt_num, gt_h, gt_w = gt_info[1]
            lr_num, lr_h, lr_w = lr_info[1]
            s = self.scale
            if (gt_num != lr_num) or (gt_h != lr_h * s) or (gt_w != lr_w * s):
                raise ValueError(
                    f"video size mismatch ({gt_info[1]} vs. {lr_info[1]} for the {i} key)"
                )

            if gt_info[2] != lr_info[2]:
                raise ValueError(
                    f"frame mismatch ({gt_info[2]} vs. {lr_info[2]} for the {i} key)"
                )

    @staticmethod
    def init_lmdb(seq_dir: Union[str, Path]):
        """LMDBデータベースを初期化して読み取り専用で開く関数"""
        import lmdb

        seq_dir = Path(seq_dir)
        env = lmdb.open(
            str(seq_dir), readonly=True, lock=False, readahead=False, meminit=False
        )
        return env

    @staticmethod
    def parse_lmdb_key(key: str):
        """LMDBのキーを解析し、ビデオインデックス、サイズ、フレーム番号を抽出"""
        key_lst = key.split("_")
        idx, size, frm = key_lst[:-2], key_lst[-2], int(key_lst[-1])
        idx = "_".join(idx)
        size = tuple(map(int, size.split("x")))  # n_frm, h, w
        return idx, size, frm

    @staticmethod
    def read_lmdb_frame(env, key: str, size: tuple) -> np.ndarray:
        """LMDBデータベースから指定されたキーのフレームを読み取る関数"""
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode("ascii"))
        frm = np.frombuffer(buf, dtype=np.uint8).reshape(*size)
        return frm

    def crop_sequence(self, **kwargs):
        """シーケンスをクロップする（サブクラスで実装する必要あり）"""
        pass

    @staticmethod
    def augment_sequence(**kwargs):
        """シーケンスを拡張する（サブクラスで実装する必要あり）"""
        pass


class ImageFolderDataset(BaseDataset):
    """対応するペアがない（unpaired）データを扱うフォルダデータセット"""

    def __init__(self, data_opt: dict, **kwargs: Optional[dict]) -> None:
        # 親クラス（BaseDataset）の初期化を実行
        super(ImageFolderDataset, self).__init__(data_opt, **kwargs)

        # ディレクトリ内のアイテム（ファイルやディレクトリ）を取得
        all_items = list(Path(self.lr_seq_dir).iterdir())

        # ディレクトリのみをフィルタリングしてキーとして保持
        self.keys = sorted([item.name for item in all_items if item.is_dir()])

    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.keys)

    def __getitem__(self, item: int) -> dict:
        """指定されたインデックスに対応するデータを返す"""
        key = self.keys[item]

        # 画像を読み込む
        img_seq = []
        for img_path in retrieve_files(Path(self.lr_seq_dir) / key):
            img = cv2.imread(str(img_path))[..., ::-1].astype(np.float32) / 255.0
            img_seq.append(img)
        img_seq = np.stack(img_seq)  # thwc|rgb|float32

        # テンソルに変換
        img_tsr = torch.from_numpy(np.ascontiguousarray(img_seq))  # float32

        # lr: 低解像度画像シーケンス
        return {
            "lr": img_tsr,
            "seq_idx": key,
            "frm_idx": sorted(
                [p.name for p in (Path(self.lr_seq_dir) / key).iterdir()]
            ),
        }


class PairedFolderDataset(BaseDataset):
    """対応するペアがある（paired）データを扱うフォルダデータセット。BI & BDの劣化モデルをサポート"""

    def __init__(self, data_opt: dict, **kwargs: Optional[dict]) -> None:
        # 親クラス（BaseDataset）の初期化を実行
        super(PairedFolderDataset, self).__init__(data_opt, **kwargs)

        # 低解像度（LR）ディレクトリ内のキーを取得
        self.keys = sorted([p.name for p in Path(self.lr_seq_dir).iterdir()])

        # 必要に応じてキーをフィルタリング
        sel_keys = set(self.keys)
        if hasattr(self, "filter_file") and self.filter_file is not None:
            with open(self.filter_file, "r") as f:
                sel_keys = {line.strip() for line in f}
        elif hasattr(self, "filter_list") and self.filter_list is not None:
            sel_keys = set(self.filter_list)
        self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.keys)

    def __getitem__(self, item: int) -> dict:
        """指定されたインデックスに対応するデータを返す"""
        key = self.keys[item]

        # 低解像度フレームのみを読み込む
        lr_seq = []
        for frm_path in retrieve_files(Path(self.lr_seq_dir) / key):
            frm = (
                cv2.imread(str(frm_path))[..., ::-1].astype(np.float32) / 255.0
            )  # BGRからRGBに変換
            lr_seq.append(frm)
        lr_seq = np.stack(lr_seq)  # 画像シーケンスをスタック（thwc形式）

        # テンソルに変換
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq))  # float32形式

        # lr: 低解像度画像シーケンス
        return {
            "lr": lr_tsr,
            "seq_idx": key,
            "frm_idx": sorted(
                [p.name for p in (Path(self.lr_seq_dir) / key).iterdir()]
            ),
        }


class UnpairedFolderDataset(BaseDataset):
    """対応するペアがない（unpaired）データを扱うフォルダデータセット（BD劣化用）"""

    def __init__(self, data_opt: dict, **kwargs: Optional[dict]) -> None:
        # 親クラス（BaseDataset）の初期化を実行
        super(UnpairedFolderDataset, self).__init__(data_opt, **kwargs)

        # 高解像度（GT）ディレクトリ内のキーを取得
        self.keys = sorted([p.name for p in Path(self.gt_seq_dir).iterdir()])

        # 必要に応じてキーをフィルタリング
        sel_keys = set(self.keys)
        if hasattr(self, "filter_file") and self.filter_file is not None:
            with open(self.filter_file, "r") as f:
                sel_keys = {line.strip() for line in f}
        elif hasattr(self, "filter_list") and self.filter_list is not None:
            sel_keys = set(self.filter_list)
        self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.keys)

    def __getitem__(self, item: int) -> dict:
        """指定されたインデックスに対応するデータを返す"""
        key = self.keys[item]

        # 高解像度フレームを読み込む
        gt_seq = []
        for frm_path in retrieve_files(Path(self.gt_seq_dir) / key):
            gt_frm = cv2.imread(str(frm_path))[..., ::-1]  # BGRからRGBに変換
            gt_seq.append(gt_frm)
        gt_seq = np.stack(gt_seq)  # 画像シーケンスをスタック（thwc形式）

        # テンソルに変換
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8形式

        # gt: 高解像度画像シーケンス
        return {
            "gt": gt_tsr,
            "seq_idx": key,
            "frm_idx": sorted(
                [p.name for p in (Path(self.gt_seq_dir) / key).iterdir()]
            ),
        }
