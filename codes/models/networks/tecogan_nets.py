# codes\models\networks\tecogan_nets.py
"""
このコードの概要
このコードは、フレーム間の再帰型ネットワーク（FRNet）を含む生成器モジュールと、フレームの識別器（Discriminator）モジュールを定義しています。各モジュールには、入力データに対して操作を行うためのさまざまなネットワークが含まれます。以下は、このコードの各セクションの概要です：

FNet: オプティカルフロー（光の流れ）を推定するためのネットワーク。2つのフレームの間の動きの特徴を抽出し、それをアップサンプリングしてフローを推定します。
ResidualBlock: 残差ブロックを持つネットワーク。このブロックは、入力を変換し、スキップ接続で元の入力に変換を加えます。
SRNet: フレームの再構築とアップサンプリングを行うためのネットワーク。低解像度のフレームを受け取り、高解像度のフレームを出力します。
FRNet: フレーム間の再帰型ネットワークで、低解像度フレームを高解像度フレームに変換します。フレーム間のオプティカルフローを利用して高解像度化を行います。
DiscriminatorBlocks: フレームの識別器で使用される一連のブロック。入力フレームを順に処理し、特徴を抽出します。
SpatioTemporalDiscriminator: フレームの時空間的な識別器。複数のフレームを使用して、時空間的な特徴を持つフレームの分類を行います。
SpatialDiscriminator: 単一フレームに対する識別器。条件付き入力を使用する場合と使用しない場合で構成され、フレームの分類を行います。
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# モジュールのインポート
from .base_nets import BaseSequenceGenerator, BaseSequenceDiscriminator
from codes.utils.net_utils import space_to_depth, backward_warp, get_upsampling_func
from codes.utils.net_utils import initialize_weights
from codes.utils.data_utils import float32_to_uint8
from codes.metrics.model_summary import register, parse_model_info


# ====================== generator modules ====================== #
class FNet(nn.Module):
    """Optical flow estimation network"""

    def __init__(self, in_nc):
        super(FNet, self).__init__()

        # エンコーダ1: 2つの入力チャネル（前フレームと現在のフレーム）を32次元にエンコード
        self.encoder1 = nn.Sequential(
            nn.Conv2d(2 * in_nc, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # エンコーダ2: エンコーダ1の出力を64次元にエンコード
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # エンコーダ3: エンコーダ2の出力を128次元にエンコード
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # デコーダ1: 128次元から256次元へのデコード
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # デコーダ2: 256次元から128次元へのデコード
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # デコーダ3: 128次元から64次元へのデコード
        self.decoder3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # オプティカルフロー推定
        self.flow = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True),
        )

    def forward(self, x1, x2):
        """Compute optical flow from x1 to x2"""

        # エンコーダを通してフローの特徴を抽出
        out = self.encoder1(torch.cat([x1, x2], dim=1))
        out = self.encoder2(out)
        out = self.encoder3(out)
        # デコーダで特徴を復元しながら解像度をアップサンプリング
        out = F.interpolate(
            self.decoder1(out), scale_factor=2, mode="bilinear", align_corners=False
        )
        out = F.interpolate(
            self.decoder2(out), scale_factor=2, mode="bilinear", align_corners=False
        )
        out = F.interpolate(
            self.decoder3(out), scale_factor=2, mode="bilinear", align_corners=False
        )
        # オプティカルフローを推定
        out = torch.tanh(self.flow(out)) * 24  # 24 is the max velocity

        return out


class ResidualBlock(nn.Module):
    """Residual block without batch normalization"""

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()

        # 残差ブロック
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        # 入力に対して変換を適用し、元の入力に加算（スキップ接続）
        out = self.conv(x) + x
        return out


class SRNet(nn.Module):
    """Reconstruction & Upsampling network"""

    def __init__(self, in_nc, out_nc, nf, nb, upsample_func, scale):
        super(SRNet, self).__init__()

        # 入力層の定義
        self.conv_in = nn.Sequential(
            nn.Conv2d((scale**2 + 1) * in_nc, nf, 3, 1, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        # 残差ブロックの定義
        self.resblocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # アップサンプリング層の定義
        conv_up = [
            nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
        ]

        if scale == 4:
            # スケールが4の場合、追加のアップサンプリング層を追加
            conv_up += [
                nn.ConvTranspose2d(nf, nf, 3, 2, 1, output_padding=1, bias=True),
                nn.ReLU(inplace=True),
            ]

        self.conv_up = nn.Sequential(*conv_up)

        # 出力層の定義
        self.conv_out = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # アップサンプリング関数
        self.upsample_func = upsample_func

    def forward(self, lr_curr, hr_prev_tran):
        """lr_curr: the current lr data in shape nchw
        hr_prev_tran: the previous transformed hr_data in shape n(s*s*c)hw
        """

        # 入力データと前のフレームを結合し、ネットワークに入力
        out = self.conv_in(torch.cat([lr_curr, hr_prev_tran], dim=1))
        # 残差ブロックを通過
        out = self.resblocks(out)
        # アップサンプリング
        out = self.conv_up(out)
        # 出力層
        out = self.conv_out(out)
        # 入力データをアップサンプリングし、結果に加算
        out += self.upsample_func(lr_curr)

        return out


class FRNet(BaseSequenceGenerator):
    """Frame-recurrent network: https://arxiv.org/abs/1801.04590"""

    def __init__(self, in_nc, out_nc, nf, nb, degradation, scale):
        super(FRNet, self).__init__()

        self.scale = scale

        # 劣化タイプに基づいたアップサンプリング関数の取得
        self.upsample_func = get_upsampling_func(self.scale, degradation)

        # FNetとSRNetの定義
        self.fnet = FNet(in_nc)
        self.srnet = SRNet(in_nc, out_nc, nf, nb, self.upsample_func, self.scale)

    def forward(self, lr_data, device=None):
        # 訓練モードか推論モードかで処理を分ける
        if self.training:
            out = self.forward_sequence(lr_data)
        else:
            out = self.infer_sequence(lr_data, device)

        return out

    def forward_sequence(self, lr_data):
        """
        Parameters:
            :param lr_data: lr data in shape ntchw
        """

        # 入力データの次元取得
        n, t, c, lr_h, lr_w = lr_data.size()
        hr_h, hr_w = lr_h * self.scale, lr_w * self.scale

        # オプティカルフローの計算
        lr_prev = lr_data[:, :-1, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_curr = lr_data[:, 1:, ...].reshape(n * (t - 1), c, lr_h, lr_w)
        lr_flow = self.fnet(lr_curr, lr_prev)  # n*(t-1),2,h,w

        # 低解像度のフローをアップサンプリング
        hr_flow = self.scale * self.upsample_func(lr_flow)
        hr_flow = hr_flow.view(n, (t - 1), 2, hr_h, hr_w)

        # 最初の高解像度データの計算
        hr_data = []
        hr_prev = self.srnet(
            lr_data[:, 0, ...],
            torch.zeros(
                n,
                (self.scale**2) * c,
                lr_h,
                lr_w,
                dtype=torch.float32,
                device=lr_data.device,
            ),
        )
        hr_data.append(hr_prev)

        # 残りの高解像度データの計算
        for i in range(1, t):
            # 前のフレームをワープ
            hr_prev_warp = backward_warp(hr_prev, hr_flow[:, i - 1, ...])

            # 現在のフレームの高解像度化
            hr_curr = self.srnet(
                lr_data[:, i, ...], space_to_depth(hr_prev_warp, self.scale)
            )

            # データを保存および更新
            hr_data.append(hr_curr)
            hr_prev = hr_curr

        hr_data = torch.stack(hr_data, dim=1)  # n,t,c,hr_h,hr_w

        # 出力辞書を構築
        ret_dict = {
            "hr_data": hr_data,  # n,t,c,hr_h,hr_w
            "hr_flow": hr_flow,  # n,t,2,hr_h,hr_w
            "lr_prev": lr_prev,  # n(t-1),c,lr_h,lr_w
            "lr_curr": lr_curr,  # n(t-1),c,lr_h,lr_w
            "lr_flow": lr_flow,  # n(t-1),2,lr_h,lr_w
        }

        return ret_dict

    def step(self, lr_curr, lr_prev, hr_prev):
        """
        Parameters:
            :param lr_curr: the current lr data in shape nchw
            :param lr_prev: the previous lr data in shape nchw
            :param hr_prev: the previous hr data in shape nc(sh)(sw)
        """

        # 低解像度のフローを推定
        lr_flow = self.fnet(lr_curr, lr_prev)

        # サイズが8の倍数でない場合にパディング
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), "reflect")

        # 低解像度のフローをアップサンプリング
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)

        # 前のフレームをワープ
        hr_prev_warp = backward_warp(hr_prev, hr_flow)

        # 現在のフレームを高解像度化
        hr_curr = self.srnet(lr_curr, space_to_depth(hr_prev_warp, self.scale))

        return hr_curr

    def infer_sequence(self, lr_data, device):
        """
        Parameters:
            :param lr_data: torch.FloatTensor in shape tchw
            :param device: torch.device

            :return hr_seq: uint8 np.ndarray in shape tchw
        """

        # パラメータの設定
        tot_frm, c, h, w = lr_data.size()
        s = self.scale

        # 推論処理
        hr_seq = []
        lr_prev = torch.zeros(1, c, h, w, dtype=torch.float32).to(device)
        hr_prev = torch.zeros(1, c, s * h, s * w, dtype=torch.float32).to(device)

        with torch.no_grad():
            for i in range(tot_frm):
                lr_curr = lr_data[i : i + 1, ...].to(device)
                hr_curr = self.step(lr_curr, lr_prev, hr_prev)
                lr_prev, hr_prev = lr_curr, hr_curr

                hr_frm = hr_curr.squeeze(0).cpu().numpy()  # chw|rgb|uint8
                hr_seq.append(float32_to_uint8(hr_frm))

        return np.stack(hr_seq).transpose(0, 2, 3, 1)  # thwc

    def generate_dummy_data(self, lr_size, device):
        c, lr_h, lr_w = lr_size
        s = self.scale

        # ダミー入力データの生成
        lr_curr = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32).to(device)
        lr_prev = torch.rand(1, c, lr_h, lr_w, dtype=torch.float32).to(device)
        hr_prev = torch.rand(1, c, s * lr_h, s * lr_w, dtype=torch.float32).to(device)

        data_list = [lr_curr, lr_prev, hr_prev]
        return data_list

    def profile(self, lr_size, device):
        gflops_dict, params_dict = OrderedDict(), OrderedDict()

        # ダミー入力データの生成
        lr_curr, lr_prev, hr_prev = self.generate_dummy_data(lr_size, device)

        # プロファイルモジュール1: フロー推定モジュール
        lr_flow = register(self.fnet, [lr_curr, lr_prev])
        gflops_dict["FNet"], params_dict["FNet"] = parse_model_info(self.fnet)

        # プロファイルモジュール2: 超解像モジュール
        pad_h = lr_curr.size(2) - lr_curr.size(2) // 8 * 8
        pad_w = lr_curr.size(3) - lr_curr.size(3) // 8 * 8
        lr_flow_pad = F.pad(lr_flow, (0, pad_w, 0, pad_h), "reflect")
        hr_flow = self.scale * self.upsample_func(lr_flow_pad)
        hr_prev_warp = backward_warp(hr_prev, hr_flow)
        _ = register(self.srnet, [lr_curr, space_to_depth(hr_prev_warp, self.scale)])
        gflops_dict["SRNet"], params_dict["SRNet"] = parse_model_info(self.srnet)

        return gflops_dict, params_dict


# ====================== discriminator modules ====================== #
class DiscriminatorBlocks(nn.Module):
    def __init__(self):
        super(DiscriminatorBlocks, self).__init__()

        # 識別器のブロック1: 入力データを2分の1にダウンサンプリング
        self.block1 = nn.Sequential(  # /2
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 識別器のブロック2: 入力データを4分の1にダウンサンプリング
        self.block2 = nn.Sequential(  # /4
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 識別器のブロック3: 入力データを8分の1にダウンサンプリング
        self.block3 = nn.Sequential(  # /8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 識別器のブロック4: 入力データを16分の1にダウンサンプリング
        self.block4 = nn.Sequential(  # /16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # 各ブロックを順に通過
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        feature_list = [out1, out2, out3, out4]

        return out4, feature_list


class SpatioTemporalDiscriminator(BaseSequenceDiscriminator):
    """Spatio-Temporal discriminator proposed in TecoGAN"""

    def __init__(self, in_nc, spatial_size, tempo_range, degradation, scale):
        super(SpatioTemporalDiscriminator, self).__init__()

        # 基本設定
        mult = 3  # (conditional triplet, input triplet, warped triplet)
        self.spatial_size = spatial_size
        self.tempo_range = tempo_range
        assert self.tempo_range == 3, "currently only support 3 as tempo_range"
        self.scale = scale

        # 入力層
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc * tempo_range * mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 識別器ブロック
        self.discriminator_block = DiscriminatorBlocks()  # downsample 16x

        # クラス分類器
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

        # 劣化タイプに基づくアップサンプリング関数の取得
        self.upsample_func = get_upsampling_func(self.scale, degradation)

    def forward(self, data, args_dict):
        # シーケンスの順伝播
        out = self.forward_sequence(data, args_dict)
        return out

    def forward_sequence(self, data, args_dict):
        """
        :param data: should be either hr_data or gt_data
        :param args_dict: a dict including data/config required here
        """

        # === パラメータの設定 === #
        net_G = args_dict["net_G"]
        lr_data = args_dict["lr_data"]
        bi_data = args_dict["bi_data"]
        hr_flow = args_dict["hr_flow"]

        n, t, c, lr_h, lr_w = lr_data.size()
        _, _, _, hr_h, hr_w = data.size()

        s_size = self.spatial_size
        t = t // 3 * 3  # 3の倍数のフレーム数に合わせる
        n_clip = n * t // 3  # 全バッチの3フレームクリップの合計数

        c_size = int(s_size * args_dict["crop_border_ratio"])
        n_pad = (s_size - c_size) // 2

        # === フォワードフローとバックワードフローの計算 === #
        if "hr_flow_merge" not in args_dict:
            if args_dict["use_pp_crit"]:
                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # 例: frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)
                hr_flow_fw = hr_flow.flip(1)[:, 1:t:3, ...]
            else:
                lr_curr = lr_data[:, 1:t:3, ...]
                lr_curr = lr_curr.reshape(n_clip, c, lr_h, lr_w)

                lr_next = lr_data[:, 2:t:3, ...]
                lr_next = lr_next.reshape(n_clip, c, lr_h, lr_w)

                # フォワードフローの計算
                lr_flow_fw = net_G.fnet(lr_curr, lr_next)
                hr_flow_fw = self.scale * self.upsample_func(lr_flow_fw)

                hr_flow_bw = hr_flow[:, 0:t:3, ...]  # 例: frame1 -> frame0
                hr_flow_idle = torch.zeros_like(hr_flow_bw)  # frame1 -> frame1
                hr_flow_fw = hr_flow_fw.view(
                    n, t // 3, 2, hr_h, hr_w
                )  # frame1 -> frame2

            # bw/idle/fwフローのマージ
            hr_flow_merge = torch.stack(
                [hr_flow_bw, hr_flow_idle, hr_flow_fw], dim=2
            )  # n,t//3,3,2,h,w

            # 形状の変更と勾配伝搬の停止
            hr_flow_merge = hr_flow_merge.view(n_clip * 3, 2, hr_h, hr_w).detach()

        else:
            # 計算を減らすためにデータを再利用
            hr_flow_merge = args_dict["hr_flow_merge"]

        # === Dの入力データの構築（3つのパーツ） === #
        # パート1: バイキュービックアップサンプリングデータ（条件付き入力）
        cond_data = bi_data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        # 注意: ここでは順序の変更は必須ではないが、TecoGAN-Tensorflowと同じ実装にするために実施
        cond_data = cond_data.permute(0, 2, 1, 3, 4)
        cond_data = cond_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # パート2: オリジナルデータ
        orig_data = data[:, :t, ...].reshape(n_clip, 3, c, hr_h, hr_w)
        orig_data = orig_data.permute(0, 2, 1, 3, 4)
        orig_data = orig_data.reshape(n_clip, c * 3, hr_h, hr_w)

        # パート3: ワープされたデータ
        warp_data = backward_warp(
            data[:, :t, ...].reshape(n * t, c, hr_h, hr_w), hr_flow_merge
        )
        warp_data = warp_data.view(n_clip, 3, c, hr_h, hr_w)
        warp_data = warp_data.permute(0, 2, 1, 3, 4)
        warp_data = warp_data.reshape(n_clip, c * 3, hr_h, hr_w)
        # TecoGANで提案されたように、トレーニングの安定性を向上させるためにボーダーを削除
        warp_data = F.pad(
            warp_data[..., n_pad : n_pad + c_size, n_pad : n_pad + c_size],
            (n_pad,) * 4,
            mode="constant",
        )

        # 3つのパーツを結合
        input_data = torch.cat([orig_data, warp_data, cond_data], dim=1)

        # === クラス分類 === #
        out = self.conv_in(input_data)
        out, feature_list = self.discriminator_block(out)
        out = out.view(out.size(0), -1)
        out = self.dense(out)
        pred = out, feature_list

        # 出力辞書の構築（pred以外のデータも返す）
        ret_dict = {"hr_flow_merge": hr_flow_merge}

        return pred, ret_dict


class SpatialDiscriminator(BaseSequenceDiscriminator):
    """Spatial discriminator"""

    def __init__(self, in_nc, spatial_size, use_cond):
        super(SpatialDiscriminator, self).__init__()

        # 基本設定
        self.use_cond = use_cond  # 条件付き入力を使用するかどうか
        mult = 2 if self.use_cond else 1
        tempo_range = 1

        # 入力層
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_nc * tempo_range * mult, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 識別器ブロック
        self.discriminator_block = DiscriminatorBlocks()  # /16

        # クラス分類器
        self.dense = nn.Linear(256 * spatial_size // 16 * spatial_size // 16, 1)

    def forward(self, data, args_dict):
        # シーケンスの順伝播
        out = self.forward_sequence(data, args_dict)
        return out

    def step(self, x):
        # 各ブロックを順に通過
        out = self.conv_in(x)
        out, feature_list = self.discriminator_block(out)

        out = out.view(out.size(0), -1)
        out = self.dense(out)

        return out, feature_list

    def forward_sequence(self, data, args_dict):
        # === パラメータの設定 === #
        n, t, c, hr_h, hr_w = data.size()
        data = data.view(n * t, c, hr_h, hr_w)

        # === net_Dの入力データの構築 === #
        if self.use_cond:
            bi_data = args_dict["bi_data"].view(n * t, c, hr_h, hr_w)
            input_data = torch.cat([bi_data, data], dim=1)
        else:
            input_data = data

        # === クラス分類 === #
        pred = self.step(input_data)

        # 出力辞書の構築（返すデータなし）
        ret_dict = {}

        return pred, ret_dict
