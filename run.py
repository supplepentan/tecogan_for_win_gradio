# run.py
from pathlib import Path
import shutil
from config import INPUT_DIR, INPUT_IMAGES_FOLDER, INPUT_MOVIE_FILENAME
from config import OUTPUT_DIR, OUTPUT_IMAGES_FOLDER, OUTPUT_MOVIE_FILENAME
from codes.utils import base_utils, movie_utils, inference_utils

input_dir = Path(INPUT_DIR)
input_images_folder = Path(INPUT_IMAGES_FOLDER)
input_movie_filename = Path(INPUT_MOVIE_FILENAME)
output_dir = Path(OUTPUT_DIR)
output_images_folder = Path(OUTPUT_IMAGES_FOLDER)
output_movie_filename = Path(OUTPUT_MOVIE_FILENAME)


def clean_output_directory():
    """出力ディレクトリをクリーンアップします。"""
    output_images_path = output_dir.joinpath(output_images_folder)
    output_movie_path = output_dir.joinpath(output_movie_filename)

    if output_images_path.exists():
        shutil.rmtree(output_images_path)

    if output_movie_path.exists():
        output_movie_path.unlink()


def super_resolution():
    """ビデオ処理の主要な手順を実行します。"""
    input_video_path = INPUT_DIR.joinpath(INPUT_MOVIE_FILENAME)
    input_images_path = INPUT_DIR.joinpath(INPUT_IMAGES_FOLDER)
    output_video_path = output_dir.joinpath(output_movie_filename)
    output_images_path = output_dir.joinpath(output_images_folder)

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
