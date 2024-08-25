import gradio as gr
from pathlib import Path
import shutil
from typing import List

from config import (
    INPUT_DIRECTORYNAME,
    INPUT_IMAGES_DIRECTORYNAME,
    INPUT_MOVIE_FILENAME,
    OUTPUT_DIRECTORYNAME,
    OUTPUT_IMAGES_DIRECTORYNAME,
    OUTPUT_MOVIE_FILENAME,
    OUTPUT_AUDIO_FILENAME,
    PRETRAINED_MODELS_DIRECTORY,
)
from codes.utils import base_utils, inference_utils, movie_utils

# 入力と出力のディレクトリおよびファイル名の定義
input_directory: Path = Path(INPUT_DIRECTORYNAME)
input_images_directory: Path = input_directory.joinpath(INPUT_IMAGES_DIRECTORYNAME)
input_movie_path: Path = input_directory.joinpath(INPUT_MOVIE_FILENAME)

output_directory: Path = Path(OUTPUT_DIRECTORYNAME)
output_images_directory: Path = output_directory.joinpath(OUTPUT_IMAGES_DIRECTORYNAME)
output_movie_path: Path = output_directory.joinpath(OUTPUT_MOVIE_FILENAME)
output_audio_path: Path = output_directory.joinpath(OUTPUT_AUDIO_FILENAME)

pretrained_models_directory: Path = Path(PRETRAINED_MODELS_DIRECTORY)


# 学習済みモデルのリストを取得する関数
def get_pretrained_models() -> List[str]:
    return [model.name for model in pretrained_models_directory.iterdir()]


# アップロードされたビデオを保存する関数
def save_uploaded_video(uploaded_video: gr.File) -> Path:
    shutil.copy(uploaded_video.name, input_movie_path)
    return input_movie_path


# 出力ディレクトリのクリーンアップを行う関数
def clean_output_directory() -> None:
    if output_images_directory.exists():
        shutil.rmtree(output_images_directory)
    if output_movie_path.exists():
        output_movie_path.unlink()
    if output_audio_path.exists():
        output_audio_path.unlink()


# ビデオを処理する関数
def process_video(input_video_path: Path, model_name: str) -> Path:
    video_frame_rate: float = movie_utils.get_video_frame_rate(input_video_path)

    # 画像の抽出
    movie_utils.extract_images_from_video(
        input_video_path, input_images_directory, framerate=video_frame_rate
    )

    # オーディオの抽出
    movie_utils.extract_audio_from_video(input_video_path, output_audio_path)

    # モデル推論の実行
    opt = base_utils.opt(model_name=model_name)
    inference_utils.inference(opt)

    # 画像からビデオの作成とオーディオの追加
    movie_utils.create_video_from_images(
        output_images_directory,
        output_movie_path,
        audio_path=output_audio_path,
        framerate=video_frame_rate,
    )

    return output_movie_path


# メイン関数
def run(input_video_file: gr.File, model_name: str) -> Path:
    input_video_path: Path = save_uploaded_video(input_video_file)
    clean_output_directory()
    return process_video(input_video_path, model_name)


# Gradioインターフェースの設定
css: str = """
    .submit_button_class { height: 50px; }
"""
with gr.Blocks(css=css) as demo:
    gr.Markdown("### TecoGAN")
    with gr.Row():
        with gr.Column():
            input_video_preview = gr.Video()
        with gr.Column():
            output_video = gr.Video()
    with gr.Row():
        input_video = gr.File(label="アップロードする動画")
        model_selector = gr.Dropdown(
            label="モデルを選択", choices=get_pretrained_models()
        )
    with gr.Row():
        submit_button = gr.Button("超解像", elem_id="submit_button_class")

    # ファイルアップロード時にプレビューを表示
    input_video.change(fn=lambda x: x, inputs=input_video, outputs=input_video_preview)
    submit_button.click(
        fn=run,
        inputs=[input_video, model_selector],
        outputs=output_video,
    )

demo.launch()
