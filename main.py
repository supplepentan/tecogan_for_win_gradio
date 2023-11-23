import gradio as gr
import os
import shutil
import ffmpeg

from codes.utils import base_utils, inference_utils, movie_utils

# 定数の定義
INPUT_DIR = "data"
INPUT_IMAGES_FOLDER = "input"
INPUT_MOVIE_FILENAME = "input.mp4"

OUTPUT_DIR = "results"
OUTPUT_IMAGES_FOLDER = "output_images"
OUTPUT_MOVIE_FILENAME = "output.mp4"
OUTPUT_AUDIO_FILENAME = "output_audio.mp3"


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
    output_images_folder = os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER)
    output_movie_path = os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)
    output_audio_path = os.path.join(OUTPUT_DIR, OUTPUT_AUDIO_FILENAME)

    if os.path.exists(output_images_folder):
        shutil.rmtree(output_images_folder)
    if os.path.exists(output_movie_path):
        os.remove(output_movie_path)
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)


def process_video(input_video_path, model_name):
    video_frame_rate = movie_utils.get_video_frame_rate(input_video_path)

    movie_utils.extract_images_from_video(
        input_video_path,
        os.path.join(INPUT_DIR, INPUT_IMAGES_FOLDER),
        framerate=video_frame_rate,
    )

    movie_utils.extract_audio_from_video(
        input_video_path, os.path.join(OUTPUT_DIR, OUTPUT_AUDIO_FILENAME)
    )

    opt = base_utils.opt(model_name=model_name)
    inference_utils.inference(opt)

    movie_utils.create_video_from_images(
        os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER),
        os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME),
        audio_path=os.path.join(OUTPUT_DIR, OUTPUT_AUDIO_FILENAME),
        framerate=video_frame_rate,
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
