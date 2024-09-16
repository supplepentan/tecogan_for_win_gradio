from pathlib import Path
from codes.utils import base_utils, inference_utils, movie_utils


class TecoGanProcessor:
    def __init__(
        self,
        input_movie,
        output_directory,
        pretrained_model,
    ):
        self.input_movie_path = Path(input_movie)
        self.output_directory_path = Path(output_directory)
        self.pretrained_model_path = Path(pretrained_model)
        self.extracted_images_directory_path = Path(output_directory).joinpath(
            "extracted_images"
        )
        self.extracted_audio_path = Path(output_directory).joinpath(
            "extracted_audio.mp3"
        )
        self.resolved_images_directory_path = Path(output_directory).joinpath(
            "resolved_images"
        )
        self.resolved_movie_file_path = Path(output_directory).joinpath(
            "output_movie.mp4"
        )
        self.print_condition()
        self.process()

    def print_condition(self):
        print("=" * 40)
        print("input movie path:", self.input_movie_path)
        print("output directory path:", self.output_directory_path)
        print("pretrained model path:", self.pretrained_model_path)
        print("extracted images directory path:", self.extracted_images_directory_path)
        print("extracted audio path:", self.extracted_audio_path)
        print("resolved images directory ath:", self.resolved_images_directory_path)
        print("resolved movie file path:", self.resolved_movie_file_path)
        print("=" * 40)

    def process(self) -> Path:
        """ビデオを処理する関数"""
        # 動画のフレームレートを抽出
        video_frame_rate: float = movie_utils.get_video_frame_rate(
            str(self.input_movie_path)
        )

        # 動画から画像の抽出
        movie_utils.extract_images_from_video(
            str(self.input_movie_path),
            str(self.extracted_images_directory_path),
            framerate=video_frame_rate,
        )

        # 画像ファイルが存在するか確認
        images = list(Path(self.extracted_images_directory_path).glob("*.png"))
        if len(images) == 0:
            raise FileNotFoundError(
                f"No images found in directory {self.extracted_images_directory_path}"
            )

        movie_utils.extract_audio_from_video(
            str(self.input_movie_path), str(self.extracted_audio_path)
        )

        opt = base_utils.opt(
            str(self.output_directory_path),
            str(self.resolved_images_directory_path),
            str(self.pretrained_model_path),
        )
        inference_utils.inference(opt)

        movie_utils.create_video_from_images(
            str(self.resolved_images_directory_path),
            str(self.resolved_movie_file_path),
            audio_path=str(self.extracted_audio_path),
            framerate=video_frame_rate,
        )
        return
