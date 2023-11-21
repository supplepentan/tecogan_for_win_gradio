import os
import shutil

from codes.utils import base_utils, movie_utils, inference_utils

# 定数の定義
INPUT_DIR = "data"
INPUT_IMAGES_FOLDER = "input"
INPUT_MOVIE_FILENAME = "input.mp4"

OUTPUT_DIR = "results"
OUTPUT_IMAGES_FOLDER = "output_images"
OUTPUT_MOVIE_FILENAME = "output.mp4"


def clean_output_directory():
    """出力ディレクトリをクリーンアップします。"""
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER)):
        shutil.rmtree(os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER))
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)):
        os.remove(os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME))


def super_resolution():
    """ビデオ処理の主要な手順を実行します。"""
    input_video_path = os.path.join(INPUT_DIR, INPUT_MOVIE_FILENAME)
    input_images_path = os.path.join(INPUT_DIR, INPUT_IMAGES_FOLDER)
    output_video_path = os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)
    output_images_path = os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER)

    # ビデオから画像への変換
    movie_utils.extract_images_from_video(input_video_path, input_images_path)

    # 機械学習モデルによる推論処理
    opt = base_utils.opt()
    inference_utils.inference(opt)

    # 画像からビデオへの変換
    movie_utils.create_video_from_images(output_images_path, output_video_path)


def main():
    """メイン関数：ビデオ処理の実行"""
    # 出力ディレクトリのクリーンアップ
    clean_output_directory()

    # ビデオ処理
    super_resolution()


if __name__ == "__main__":
    main()
