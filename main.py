import gradio as gr
import os
import shutil

from codes.utils import base_utils, movie_utils, inference_utils

INPUT_DIR = "data"
INPUT_IMAGES_FOLDER = "input"
INPUT_MOVIE_FILENAME = "input.mp4"

OUTPUT_DIR = "results"
OUTPUT_IMAGES_FOLDER = "output_images"
OUTPUT_MOVIE_FILENAME = "output.mp4"


def run(input_video_file):
    # アップロードされたファイルをINPUTフォルダに保存
    input_video_path = os.path.join(INPUT_DIR, INPUT_MOVIE_FILENAME)
    shutil.copy(input_video_file, input_video_path)
    movie_utils.extract_images_from_video(
        INPUT_MOVIE_FILENAME, os.path.join(INPUT_DIR, INPUT_IMAGES_FOLDER)
    )
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER)):
        shutil.rmtree(os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER))
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)):
        os.remove(os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME))
    opt = base_utils.opt()
    inference_utils.inference(opt)

    movie_utils.create_video_from_images(
        os.path.join(OUTPUT_DIR, OUTPUT_IMAGES_FOLDER),
        os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME),
    )
    return os.path.join(OUTPUT_DIR, OUTPUT_MOVIE_FILENAME)


with gr.Blocks() as demo:
    gr.Markdown("### TecoGAN")
    with gr.Row():
        with gr.Column():
            input_video = gr.File(label="アップロードする動画")
            submit_button = gr.Button("超解像")
        with gr.Column():
            output_video = gr.Video()

        submit_button.click(
            fn=run,
            inputs=[input_video],
            outputs=output_video,
        )

demo.launch()
