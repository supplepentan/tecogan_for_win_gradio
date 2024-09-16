# main.py
import gradio as gr
from omegaconf import OmegaConf
from pathlib import Path
import shutil
from typing import List
from pathlib import Path
from codes.utils.teco_gan_utils import TecoGanProcessor
from codes.utils.directory_and_file_utils import remove_path


def get_pretrained_models(config) -> List[str]:
    """学習済みモデルの.pthファイルのリストを取得する関数"""
    model_dir = Path(config.pretrained_models_directory)
    return [
        model.name
        for model in model_dir.iterdir()
        if model.is_file() and model.suffix == ".pth"
    ]


def clean_input_directory(input_directory) -> None:
    remove_path(input_directory)
    return


def clean_output_directory(input_directory, output_directory) -> None:
    """出力ディレクトリのクリーンアップを行います。"""
    for path in [input_directory, output_directory]:
        remove_path(path)
    return


def save_uploaded_video(
    uploaded_video, uploaded_video_save_path, movie_file_name
) -> Path:
    """アップロードされたビデオを保存する関数"""
    # 保存先ディレクトリが存在しない場合は作成
    Path(uploaded_video_save_path).mkdir(parents=True, exist_ok=True)

    # 保存先パスにファイル名を付ける
    save_path = Path(uploaded_video_save_path) / movie_file_name

    # ファイルを指定のパスにコピー
    shutil.copy(uploaded_video.name, save_path)
    return save_path


def run(input_video_file: gr.File, model_name: str, config_web) -> Path:
    clean_output_directory(
        config_web.web.input.directory_name, config_web.web.output.directory_name
    )

    # 動画を保存する際にconfigで指定されたファイル名を使用
    saved_video_path = save_uploaded_video(
        input_video_file,
        config_web.web.input.directory_name,
        config_web.web.input.movie_file_name,
    )

    # メイン関数
    processor = TecoGanProcessor(
        saved_video_path,  # 保存したビデオのパスを渡す
        config_web.web.output.directory_name,
        str(Path(config_web.pretrained_models_directory).joinpath(model_name)),
    )
    return processor.resolved_movie_file_path


def setup_gradio_interface(config):
    """Gradioインターフェースの設定"""
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
            model_selector = gr.Dropdown(
                label="モデルを選択", choices=get_pretrained_models(config)
            )
        with gr.Row():
            submit_button = gr.Button("超解像", elem_id="submit_button_class")

        input_video.change(
            fn=lambda x: x, inputs=input_video, outputs=input_video_preview
        )

        # `config`をグローバル変数として使用するか、`fn`関数の中で明示的に渡します。
        submit_button.click(
            fn=lambda video, model: run(video, model, config),
            inputs=[input_video, model_selector],
            outputs=output_video,
        )

    return demo


def boot_web():
    config_web = OmegaConf.load("config.yaml")
    demo = setup_gradio_interface(config_web)
    demo.launch()
