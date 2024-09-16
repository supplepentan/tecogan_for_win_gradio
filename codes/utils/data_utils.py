from pathlib import Path
from typing import List, Optional, Union
from scipy import signal
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def create_kernel(sigma: float, ksize: Optional[int] = None) -> torch.Tensor:
    """
    ガウスカーネルを作成する関数。

    Parameters:
        :param sigma: ガウスカーネルの標準偏差。カーネルの広がりを決定する。
        :param ksize: カーネルのサイズ。指定しない場合、sigmaに基づいて自動的に計算される。

    Returns:
        :return: ガウスカーネルを含む3チャンネルのテンソル。RGBの各チャンネルに対応。
    """
    if ksize is None:
        ksize = 1 + 2 * int(sigma * 3.0)

    gkern1d = signal.gaussian(ksize, std=sigma).reshape(ksize, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    gaussian_kernel = gkern2d / gkern2d.sum()
    zero_kernel = np.zeros_like(gaussian_kernel)

    kernel = np.float32(
        [
            [gaussian_kernel, zero_kernel, zero_kernel],
            [zero_kernel, gaussian_kernel, zero_kernel],
            [zero_kernel, zero_kernel, gaussian_kernel],
        ]
    )

    return torch.from_numpy(kernel)


def downsample_bd(
    data: torch.Tensor, kernel: torch.Tensor, scale: int, pad_data: bool
) -> torch.Tensor:
    """
    画像をダウンサンプリングする関数。

    Parameters:
        :param data: ダウンサンプリングする画像データ。torch.FloatTensor形式、形状は [nchw]。
        :param kernel: ダウンサンプリングに使用するカーネル。
        :param scale: ダウンサンプリングスケール（倍率）。
        :param pad_data: パディングを行うかどうかのフラグ。

    Returns:
        :return: ダウンサンプリングされた画像データ。
    """
    if pad_data:
        kernel_h, kernel_w = kernel.shape[-2:]
        pad_h, pad_w = kernel_h - 1, kernel_w - 1
        pad_t, pad_b = pad_h // 2, pad_h - pad_h // 2
        pad_l, pad_r = pad_w // 2, pad_w - pad_w // 2
        data = F.pad(data, (pad_l, pad_r, pad_t, pad_b), "reflect")

    return F.conv2d(data, kernel, stride=scale, bias=None, padding=0)


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """
    RGB画像をYCbCrフォーマットに変換する関数。

    Parameters:
        :param img: RGB画像 (np.uint8形式)。

    Returns:
        :return: YCbCrフォーマットに変換された画像 (np.uint8形式)。
    """
    T = np.array(
        [
            [0.256788235294118, -0.148223529411765, 0.439215686274510],
            [0.504129411764706, -0.290992156862745, -0.367788235294118],
            [0.097905882352941, 0.439215686274510, -0.071427450980392],
        ],
        dtype=np.float64,
    )
    O = np.array([16, 128, 128], dtype=np.float64)

    img = img.astype(np.float64)
    res = np.matmul(img, T) + O
    return res.clip(0, 255).round().astype(np.uint8)


def float32_to_uint8(inputs: np.ndarray) -> np.ndarray:
    """
    np.float32形式の配列をnp.uint8形式に変換する関数。

    Parameters:
        :param inputs: np.float32形式の配列 (NT)CHW、値範囲 [0, 1]。

    Returns:
        :return: np.uint8形式の配列 (NT)CHW、値範囲 [0, 255]。
    """
    return np.uint8(np.clip(np.round(inputs * 255), 0, 255))


def save_sequence(
    seq_dir: Union[str, Path],
    seq_data: np.ndarray,
    frm_idx_lst: Optional[List[str]] = None,
    to_bgr: bool = False,
) -> None:
    """
    画像シーケンスを指定されたディレクトリに保存する関数。

    Parameters:
        :param seq_dir: 画像を保存するディレクトリ。
        :param seq_data: 保存する画像シーケンスデータ、形状は thwc|uint8。
        :param frm_idx_lst: 各フレームのファイル名リスト（指定がない場合はデフォルトの連番を使用）。
        :param to_bgr: Trueの場合、画像をRGBからBGRに変換して保存。

    Returns:
        なし
    """
    seq_dir = Path(seq_dir)

    if to_bgr:
        seq_data = seq_data[..., ::-1]

    tot_frm = len(seq_data)
    if frm_idx_lst is None:
        frm_idx_lst = [f"{i:04d}.png" for i in range(tot_frm)]

    seq_dir.mkdir(parents=True, exist_ok=True)
    for i in range(tot_frm):
        print(str(seq_dir / frm_idx_lst[i]))
        cv2.imwrite(str(seq_dir / frm_idx_lst[i]), seq_data[i])
