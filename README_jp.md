# TecoGAN-PyTorch for Windows (日本語版)

### はじめに

これは、Windows 環境でのビデオ超解像 (VSR) のための **TecoGAN** (Temporally Coherent GAN) の再実装であり、Gradio を使用した推論機能のみを持つブラウザアプリです。詳細については、公式の TensorFlow 実装 [TecoGAN-TensorFlow](https://github.com/thunil/TecoGAN) および [TecoGAN-PyTorch](https://github.com/skycrapers/TecoGAN-PyTorch) を参照してください。

### 更新履歴

- 2023 年 11 月: Windows 環境で Gradio を使用した TecoGAN をリリース。
- 2024 年 9 月: コマンドモードとウェブモードを追加。
- 2025 年 6 月: トレーニングモジュールを追加。

#### オリジナル TecoGAN-PyTorch

- 2021 年 11 月: 2 倍超解像をサポート。
- 2021 年 10 月: [REDS](https://seungjunnah.github.io/Datasets/reds.html) データセットでのモデルトレーニング/テストをサポート。
- 2021 年 7 月: マルチ GPU トレーニングとテストをサポートするためにコードベースをアップグレード。

## 依存関係

- Windows
- NVIDIA GPU + CUDA
- Python >= 3.10
- PyTorch >= 2.0

## セットアップ

### 1. 事前学習済み TecoGAN モデルのダウンロード

モデルをダウンロードし、`./pretrained_models` の下に配置してください。

[BD-4x-Vimeo]
[BI-4x-Vimeo]
[BD-4x-REDS]
[BD-2x-REDS]

これらのタグは、TecoGAN のトレーニングまたはテスト中に使用される異なる設定を表します。各タグは、使用されるデータセット、適用される劣化モデル、および超解像のスケールを指定します。

**[BD-4x-Vimeo]**: この設定では、Vimeo データセットが BD (Blur and Downsample) 劣化モデルで使用され、4 倍の超解像が適用されます。

**[BI-4x-Vimeo]**: この設定では、Vimeo データセットが BI (Bicubic Interpolation) 劣化モデルで使用され、4 倍の超解像が適用されます。

**[BD-4x-REDS]**: この設定では、REDS データセットが BD 劣化モデルで使用され、4 倍の超解像が適用されます。

**[BD-2x-REDS]**: この設定では、REDS データセットが BD 劣化モデルで使用され、2 倍の超解像が適用されます。

これらの設定は、特定のタスクや要件に応じて選択できます。たとえば、4 倍の超解像が必要で、予想される劣化がぼかしとダウンサンプルである場合、[BD-4x-Vimeo] または [BD-4x-REDS] の設定が適しています。特定のデータセット (Vimeo または REDS) でのモデルのパフォーマンスを評価する場合は、それぞれのデータセット設定が選択されます。

### 2. Gradio の実行

```bash
python main.py
```

## 謝辞

このコードは、TecoGAN-PyTorch、TecoGAN-TensorFlow、BasicSR、および LPIPS を基に構築されています。コードを共有してくださった著者に感謝いたします。

## トレーニング

**注意:** VimeoTecoGAN データセットへのアクセスが困難なため、REDS などの他の公開データセットをモデルトレーニングに使用することをお勧めします。REDS をトレーニングデータセットとして使用するには、こちら からダウンロードし、以下の `VimeoTecoGAN` を `REDS` に置き換えるだけです。

1. TecoGAN-TensorFlow の指示に従って公式のトレーニングデータセットをダウンロードし、`VimeoTecoGAN/Raw` に名前を変更して、`training_module/data` の下に配置します。

2. IO を高速化するために GT データ用の LMDB を生成します。LR の対応物はトレーニング中にオンザフライで生成されます。

```bash
python training_module/scripts/create_lmdb.py --dataset VimeoTecoGAN --raw_dir training_module/data/VimeoTecoGAN/Raw --lmdb_dir training_module/data/VimeoTecoGAN/GT.lmdb
```

以下の図は、上記の 2 つのステップを完了した後のデータセット構造を示しています。

```tex
data
  ├─ VimeoTecoGAN
    ├─ Raw                 # 生データセット
      ├─ scene_2000
        └─ ***.png
      ├─ scene_2001
        └─ ***.png
      └─ ...
    └─ GT.lmdb             # LMDBデータセット
      ├─ data.mdb
      ├─ lock.mdb
      └─ meta_info.pkl     # 各キーの形式: [vid]_[total_frame]x[h]x[w]_[i-th_frame]
```

3. **(オプション、このステップは BI 劣化にのみ必要です)** Matlab の imresize 関数を使用して LR シーケンスを手動で生成し、それからそれらの LMDB を作成します。

```bash
# 生のLRビデオシーケンスを生成します。結果は training_module/data/VimeoTecoGAN/Bicubic4xLR に保存されます。
matlab -nodesktop -nosplash -r "cd training_module/scripts; generate_lr_bi"

# LRビデオシーケンスのLMDBを作成します。
python training_module/scripts/create_lmdb.py --dataset VimeoTecoGAN --raw_dir training_module/data/VimeoTecoGAN/Bicubic4xLR --lmdb_dir training_module/data/VimeoTecoGAN/Bicubic4xLR.lmdb
```

4. まず FRVSR モデルをトレーニングします。これは、その後の TecoGAN トレーニングのためのより良い初期化を提供できます。FRVSR は TecoGAN と同じジェネレーターを持っていますが、知覚トレーニング（GAN および知覚損失）はありません。

```bash
bash training_module/train.sh BD FRVSR/FRVSR_VimeoTecoGAN_4xSR_2GPU
```

> スクラッチからトレーニングする代わりに、事前学習済みの FRVSR モデルをダウンロードして使用できます。
> [BD-4x-Vimeo] [BI-4x-Vimeo] [BD-4x-REDS][BD-2x-REDS]

トレーニングが完了したら、`training_module/experiments_BD/TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU/train.yml` のジェネレーターの `load_path` を FRVSR モデルの最新のチェックポイントの重みに設定します。

5. TecoGAN モデルをトレーニングします。`training_module/train.sh` で使用する GPU を指定できます。デフォルトでは、トレーニングはバックグラウンドで実行され、出力情報は `training_module/experiments_BD/TecoGAN/TecoGAN_VimeoTecoGAN/train/train.log` にログとして記録されます。

```bash
bash training_module/train.sh BD TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU
```

6. 以下のスクリプトを実行して、トレーニングプロセスを監視し、検証パフォーマンスを視覚化します。

```bash
python training_module/scripts/monitor_training.py -dg BD -m TecoGAN/TecoGAN_VimeoTecoGAN_4xSR_2GPU -ds Vid4
```

> 検証結果は、メトリックの実装が異なるため、上記のテスト結果と完全に同じではないことに注意してください。違いは、クロッピングポリシー、LPIPS バージョン、およびその他の問題によって引き起こされます。

<p align = "center">
    <img src="training_module/resources/losses.png" width="1080" />
    <img src="training_module/resources/metrics.png" width="1080" />
</p>

### カスタムビデオでのトレーニング (簡易版)

以下の手順は、上記で説明した LMDB/Matlab ベースの方法の代替として、独自のビデオファイルでモデルをトレーニングするための簡略化された Python のみのワークフローを提供します。この方法は、私たちが改良したカスタムスクリプトに依存しています。

**前提条件:**

- 以前の議論からのすべてのスクリプト変更が適用されていることを確認してください。
- **注意:** このセクションのすべてのコマンドは、プロジェクトのルートディレクトリ (`d:\projects-d\tecogan_for_win_gradio`) から実行されることを想定しています。

**ステップ 1: トレーニングビデオの準備**

1.  `training_module` ディレクトリ内に `my_training_videos` という新しいフォルダを作成します。
2.  トレーニングビデオファイル (例: `.mp4`, `.mov`) を `training_module/my_training_videos` フォルダに配置します。

**ステップ 2: 画像データセットの生成**

これらのスクリプトは、ビデオをペアの HR (高解像度) および LR (低解像度) 画像フレームに変換します。

1.  **HR フレームの抽出:** 以下のコマンドを実行して、ビデオを PNG 画像のシーケンスに変換します。

    ```bash
    python training_module/scripts/video_to_frames.py --video_dir training_module/my_training_videos --out_dir training_module/datasets/MyCustomDataset/HR
    ```

2.  **LR フレームの生成:** 次に、HR フレームから対応するダウンスケールされた LR 画像を作成します。

    ```bash
    python training_module/scripts/generate_lr_bd.py --hr_dir training_module/datasets/MyCustomDataset/HR --lr_dir training_module/datasets/MyCustomDataset/LR --scale 4 --sigma 1.5
    ```

**ステップ 3: `train.yml` の設定**

`train.yml` (例: `training_module/options/train/my_custom_train.yml`) がカスタムデータセットローダー (`MyPairedFolder`) を使用するように設定され、ステップ 2 で作成されたフォルダを指していることを確認してください。

```yaml
dataset:
  degradation:
    type: BD

  train:
    name: MyPairedFolder
    hr_root: training_module/datasets/MyCustomDataset/HR
    lr_root: training_module/datasets/MyCustomDataset/LR
    # ... その他の設定
```

**ステップ 4: トレーニングの開始**

データセットと設定の準備ができたら、トレーニングプロセスを開始します。

```bash
python training_module/codes/main.py --exp_dir training_module/experiments_BD/MyModel --mode train --opt training_module/options/train/my_custom_train.yml --gpu_ids 0
```

トレーニングされたモデルのチェックポイントは、`training_module/experiments_BD/MyModel/train/ckpt/` ディレクトリに保存されます。
