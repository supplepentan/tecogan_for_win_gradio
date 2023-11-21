import os
import os.path as osp
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def retrieve_files(dir, suffix="png|jpg"):
    """retrive files with specific suffix under dir and sub-dirs recursively"""

    def retrieve_files_recursively(dir, file_lst):
        for d in sorted(os.listdir(dir)):
            dd = osp.join(dir, d)

            if osp.isdir(dd):
                retrieve_files_recursively(dd, file_lst)
            else:
                if osp.splitext(d)[-1].lower() in ["." + s for s in suffix]:
                    file_lst.append(dd)

    if not dir:
        return []

    if isinstance(suffix, str):
        suffix = suffix.split("|")

    file_lst = []
    retrieve_files_recursively(dir, file_lst)
    file_lst.sort()

    return file_lst


class BaseDataset(Dataset):
    def __init__(self, data_opt, **kwargs):
        # dict to attr
        for kw, args in data_opt.items():
            setattr(self, kw, args)

        # can be used to override options defined in data_opt
        for kw, args in kwargs.items():
            setattr(self, kw, args)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def check_info(self, gt_keys, lr_keys):
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
    def init_lmdb(seq_dir):
        env = lmdb.open(
            seq_dir, readonly=True, lock=False, readahead=False, meminit=False
        )
        return env

    @staticmethod
    def parse_lmdb_key(key):
        key_lst = key.split("_")
        idx, size, frm = key_lst[:-2], key_lst[-2], int(key_lst[-1])
        idx = "_".join(idx)
        size = tuple(map(int, size.split("x")))  # n_frm, h, w
        return idx, size, frm

    @staticmethod
    def read_lmdb_frame(env, key, size):
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode("ascii"))
        frm = np.frombuffer(buf, dtype=np.uint8).reshape(*size)
        return frm

    def crop_sequence(self, **kwargs):
        pass

    @staticmethod
    def augment_sequence(**kwargs):
        pass


class ImageFolderDataset(BaseDataset):
    """Folder dataset for unpaired data."""

    def __init__(self, data_opt, **kwargs):
        super(ImageFolderDataset, self).__init__(data_opt, **kwargs)

        # ディレクトリ内のファイル/ディレクトリを取得
        all_items = os.listdir(self.lr_seq_dir)

        # ディレクトリのみをフィルタリング
        self.keys = sorted(
            [
                item
                for item in all_items
                if os.path.isdir(os.path.join(self.lr_seq_dir, item))
            ]
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load images
        img_seq = []
        for img_path in retrieve_files(osp.join(self.lr_seq_dir, key)):
            img = cv2.imread(img_path)[..., ::-1].astype(np.float32) / 255.0
            img_seq.append(img)
        img_seq = np.stack(img_seq)  # thwc|rgb|float32

        # convert to tensor
        img_tsr = torch.from_numpy(np.ascontiguousarray(img_seq))  # float32

        # lr: thwc|rgb|float32
        return {
            "lr": img_tsr,
            "seq_idx": key,
            "frm_idx": sorted(os.listdir(osp.join(self.lr_seq_dir, key))),
        }


class PairedFolderDataset(BaseDataset):
    """Folder dataset for paired data. It supports both BI & BD degradation."""

    def __init__(self, data_opt, **kwargs):
        super(PairedFolderDataset, self).__init__(data_opt, **kwargs)

        # get keys from LR directory only
        self.keys = sorted(os.listdir(self.lr_seq_dir))

        # filter keys (if necessary)
        sel_keys = set(self.keys)
        if hasattr(self, "filter_file") and self.filter_file is not None:
            with open(self.filter_file, "r") as f:
                sel_keys = {line.strip() for line in f}
        elif hasattr(self, "filter_list") and self.filter_list is not None:
            sel_keys = set(self.filter_list)
        self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load lr frames only
        lr_seq = []
        for frm_path in retrieve_files(osp.join(self.lr_seq_dir, key)):
            frm = cv2.imread(frm_path)[..., ::-1].astype(np.float32) / 255.0
            lr_seq.append(frm)
        lr_seq = np.stack(lr_seq)  # thwc|rgb|float32

        # convert to tensor
        lr_tsr = torch.from_numpy(np.ascontiguousarray(lr_seq))  # float32

        # lr: thwc|rgb|float32
        return {
            "lr": lr_tsr,
            "seq_idx": key,
            "frm_idx": sorted(os.listdir(osp.join(self.lr_seq_dir, key))),
        }


class UnpairedFolderDataset(BaseDataset):
    """Folder dataset for unpaired data (for BD degradation)"""

    def __init__(self, data_opt, **kwargs):
        super(UnpairedFolderDataset, self).__init__(data_opt, **kwargs)

        # get keys
        self.keys = sorted(os.listdir(self.gt_seq_dir))

        # filter keys
        sel_keys = set(self.keys)
        if hasattr(self, "filter_file") and self.filter_file is not None:
            with open(self.filter_file, "r") as f:
                sel_keys = {line.strip() for line in f}
        elif hasattr(self, "filter_list") and self.filter_list is not None:
            sel_keys = set(self.filter_list)
        self.keys = sorted(list(sel_keys & set(self.keys)))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]

        # load gt frames
        gt_seq = []
        for frm_path in retrieve_files(osp.join(self.gt_seq_dir, key)):
            gt_frm = cv2.imread(frm_path)[..., ::-1]
            gt_seq.append(gt_frm)
        gt_seq = np.stack(gt_seq)  # thwc|rgb|uint8

        # convert to tensor
        gt_tsr = torch.from_numpy(np.ascontiguousarray(gt_seq))  # uint8

        # gt: thwc|rgb|uint8
        return {
            "gt": gt_tsr,
            "seq_idx": key,
            "frm_idx": sorted(os.listdir(osp.join(self.gt_seq_dir, key))),
        }
