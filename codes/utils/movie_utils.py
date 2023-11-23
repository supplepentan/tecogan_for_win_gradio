import os
import ffmpeg


def get_video_frame_rate(video_path):
    probe = ffmpeg.probe(video_path)
    video_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    ]
    frame_rate = eval(video_streams[0]["avg_frame_rate"])
    return frame_rate


def extract_images_from_video(
    video_path, output_dir, filename_pattern="%04d.png", framerate=None
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_pattern = os.path.join(output_dir, filename_pattern)

    input_command = ffmpeg.input(video_path)
    output_kwargs = {"format": "image2", "vcodec": "png"}
    if framerate is not None:
        output_kwargs["r"] = framerate

    ffmpeg.output(input_command, output_pattern, **output_kwargs).run()


def extract_audio_from_video(video_path, output_path):
    """ビデオから音源を抽出する関数"""
    ffmpeg.input(video_path).output(
        output_path, format="mp3", acodec="libmp3lame"
    ).run()


def create_video_from_images(image_dir, output_path, audio_path=None, framerate=30):
    image_sequence = os.path.join(image_dir, "%04d.png")

    input_command = ffmpeg.input(image_sequence, framerate=framerate)
    output_kwargs = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "r": framerate,
    }

    if audio_path:
        ffmpeg.output(
            input_command,
            ffmpeg.input(audio_path),
            output_path,
            acodec="aac",
            strict="experimental",
            **output_kwargs
        ).run()
    else:
        ffmpeg.output(input_command, output_path, **output_kwargs).run()
