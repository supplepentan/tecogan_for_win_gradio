# main.py
from omegaconf import OmegaConf
from pathlib import Path
from typing import List
from pathlib import Path
from codes.utils.teco_gan_utils import TecoGanProcessor
from codes.utils.directory_and_file_utils import remove_path
from config import ConfigCmd


class RunCmd:
    def __init__(self, config_cmd):
        # ディレクトリのパスを設定
        self.input_directory_path = Path(config_cmd.INPUT_DIRECTORY)
        self.input_movie_path = Path(
            self.input_directory_path.joinpath(config_cmd.INPUT_MOVIE_FILE)
        )
        self.pretrained_model_path = Path(
            config_cmd.PRETRAINED_MODELS_DIRECTORY
        ).joinpath(config_cmd.PRETRAINED_MODEL)
        self.output_directory_path = Path(config_cmd.OUTPUT_DIRECTORY)

        # ディレクトリのセットアップ
        self.setup_directory()

        # TecoGanProcessorのインスタンス化と処理の開始
        processor = TecoGanProcessor(
            str(self.input_movie_path),
            str(self.output_directory_path),
            str(self.pretrained_model_path),
        )

    def setup_directory(self) -> None:
        """
        出力ディレクトリのセットアップを行う。
        既存のディレクトリがあれば削除し、新たに作成する。
        """
        # 出力ディレクトリの削除と作成
        try:
            remove_path(self.output_directory_path)
            self.output_directory_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ディレクトリのセットアップ中にエラーが発生しました: {e}")


def boot_cmd():
    config_cmd = ConfigCmd()  # クラスではなくインスタンスを作成
    RunCmd(config_cmd)
