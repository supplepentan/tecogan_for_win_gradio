import gradio as gr
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


# 学習済みモデルのリストを取得
def get_pretrained_models():
    model_list = os.listdir("pretrained_models")
    return model_list


def save_uploaded_video(input_video_file):
    """アップロードされたビデオを保存する関数"""
    input_video_path = os.path.join(INPUT_DIR, INPUT_MOVIE_FILENAME)
    shutil.copy(input_video_file, input_video_path)
    return input_video_path


def clean_output_directory():
    """出力ディレクトリのクリーンアップを行う関数"""
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER)):
        shutil.rmtree(os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER))
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)):
        os.remove(os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME))


def process_video(input_video_path, model_name):
    """ビデオ処理の主要な手順を実行する関数"""
    movie_utils.extract_images_from_video(
        input_video_path, os.path.join(INPUT_DIR, INPUT_IMAGES_FOLDER)
    )
    opt = base_utils.opt(model_name=model_name)
    inference_utils.inference(opt)
    movie_utils.create_video_from_images(
        os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER),
        os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME),
    )
    return os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)


def run(input_video_file, model_name):
    """モデル選択を含むメインの関数"""
    input_video_path = save_uploaded_video(input_video_file)
    clean_output_directory()
    return process_video(input_video_path, model_name)


# Gradioインターフェースの設定
css = """
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
        model_selector = gr.Dropdown(label="モデルを選択", choices=get_pretrained_models())
    with gr.Row():
        submit_button = gr.Button("超解像", elem_id="submit_button_class")

    input_video.change(fn=lambda x: x, inputs=input_video, outputs=input_video_preview)
    submit_button.click(
        fn=run,
        inputs=[input_video, model_selector],
        outputs=output_video,
    )

demo.launch()
