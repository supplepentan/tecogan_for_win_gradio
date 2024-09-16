# movie_utils.py
from pathlib import Path
from typing import Optional, Union
import ffmpeg


def get_video_frame_rate(video_path: Union[str, Path]) -> float:
    """
    指定されたビデオファイルのフレームレートを取得する関数。

    Parameters:
        :param video_path: ビデオファイルのパス。

    Returns:
        :return: ビデオの平均フレームレート。
    """
    # ffmpeg.probeを使用してビデオファイルのメタデータを取得
    probe = ffmpeg.probe(str(video_path))

    # ビデオストリーム情報を取得
    video_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    ]

    # フレームレートの情報を評価し、計算結果を返す
    frame_rate = eval(video_streams[0]["avg_frame_rate"])
    return frame_rate


def extract_images_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    filename_pattern: str = "%04d.png",
    framerate: Optional[float] = None,
) -> None:
    """
    ビデオファイルからフレームを抽出し、画像として保存する関数。

    Parameters:
        :param video_path: ビデオファイルのパス。
        :param output_dir: 抽出した画像を保存するディレクトリ。
        :param filename_pattern: 出力される画像ファイルの命名パターン（デフォルトは "%04d.png"）。
        :param framerate: 抽出時に指定するフレームレート。指定しない場合は元のビデオのフレームレートが使用される。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / filename_pattern)

    input_command = ffmpeg.input(str(video_path))
    output_kwargs = {"format": "image2", "vcodec": "png"}

    if framerate is not None:
        output_kwargs["r"] = framerate

    ffmpeg.output(input_command, output_pattern, **output_kwargs).run()


def extract_audio_from_video(
    video_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """
    ビデオファイルから音声を抽出し、指定されたパスに保存する関数。

    Parameters:
        :param video_path: ビデオファイルのパス。
        :param output_path: 抽出された音声ファイルを保存するパス。
    """
    ffmpeg.input(str(video_path)).output(
        str(output_path), format="mp3", acodec="libmp3lame"
    ).run()


def create_video_from_images(
    image_dir: Union[str, Path],
    output_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    framerate: int = 30,
) -> None:
    """
    画像ファイル群からビデオを生成し、必要に応じて音声を追加する関数。

    Parameters:
        :param image_dir: 画像ファイルが保存されているディレクトリ。
        :param output_path: 生成されたビデオファイルを保存するパス。
        :param audio_path: 追加する音声ファイルのパス（デフォルトは None）。
        :param framerate: ビデオのフレームレート（デフォルトは30）。
    """
    image_sequence = str(Path(output_path, image_dir, "%04d.png"))
    input_command = ffmpeg.input(image_sequence, framerate=framerate)
    output_kwargs = {"vcodec": "libx264", "pix_fmt": "yuv420p", "r": framerate}

    if audio_path:
        ffmpeg.output(
            input_command,
            ffmpeg.input(str(audio_path)),
            str(output_path),
            acodec="aac",
            strict="experimental",
            **output_kwargs,
        ).run()
    else:
        ffmpeg.output(input_command, str(output_path), **output_kwargs).run()


def get_video_frame_rate(video_path: Union[str, Path]) -> float:
    """
    指定されたビデオファイルのフレームレートを取得する関数。

    Parameters:
        :param video_path: ビデオファイルのパス。

    Returns:
        :return: ビデオの平均フレームレート。
    """
    # ffmpeg.probeを使用してビデオファイルのメタデータを取得
    probe = ffmpeg.probe(str(video_path))

    # ビデオストリーム情報を取得
    video_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    ]

    # フレームレートの情報を評価し、計算結果を返す
    frame_rate = eval(video_streams[0]["avg_frame_rate"])
    return frame_rate


def extract_images_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    filename_pattern: str = "%04d.png",
    framerate: Optional[float] = None,
) -> None:
    """
    ビデオファイルからフレームを抽出し、画像として保存する関数。

    Parameters:
        :param video_path: ビデオファイルのパス。
        :param output_dir: 抽出した画像を保存するディレクトリ。
        :param filename_pattern: 出力される画像ファイルの命名パターン（デフォルトは "%04d.png"）。
        :param framerate: 抽出時に指定するフレームレート。指定しない場合は元のビデオのフレームレートが使用される。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / filename_pattern)

    input_command = ffmpeg.input(str(video_path))
    output_kwargs = {"format": "image2", "vcodec": "png"}

    if framerate is not None:
        output_kwargs["r"] = framerate

    ffmpeg.output(input_command, output_pattern, **output_kwargs).run()


def extract_audio_from_video(
    video_path: Union[str, Path], output_path: Union[str, Path]
) -> None:
    """
    ビデオファイルから音声を抽出し、指定されたパスに保存する関数。

    Parameters:
        :param video_path: ビデオファイルのパス。
        :param output_path: 抽出された音声ファイルを保存するパス。
    """
    ffmpeg.input(str(video_path)).output(
        str(output_path), format="mp3", acodec="libmp3lame"
    ).run()


def create_video_from_images(
    image_dir: Union[str, Path],
    output_path: Union[str, Path],
    audio_path: Optional[Union[str, Path]] = None,
    framerate: int = 30,
) -> None:
    """
    画像ファイル群からビデオを生成し、必要に応じて音声を追加する関数。

    Parameters:
        :param image_dir: 画像ファイルが保存されているディレクトリ。
        :param output_path: 生成されたビデオファイルを保存するパス。
        :param audio_path: 追加する音声ファイルのパス（デフォルトは None）。
        :param framerate: ビデオのフレームレート（デフォルトは30）。
    """
    # 修正点: パスの構築を修正
    image_sequence = str(Path(image_dir) / "%04d.png")

    input_command = ffmpeg.input(image_sequence, framerate=framerate)
    output_kwargs = {"vcodec": "libx264", "pix_fmt": "yuv420p", "r": framerate}

    if audio_path:
        ffmpeg.output(
            input_command,
            ffmpeg.input(str(audio_path)),
            str(output_path),
            acodec="aac",
            strict="experimental",
            **output_kwargs,
        ).run()
    else:
        ffmpeg.output(input_command, str(output_path), **output_kwargs).run()
