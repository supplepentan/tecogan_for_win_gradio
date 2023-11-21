import os
import ffmpeg


def extract_images_from_video(video_path, output_dir, filename_pattern="%04d.png"):
    """
    video_path: 分割する動画ファイルのパス
    output_dir: 画像を保存するディレクトリのパス
    filename_pattern: 出力する画像ファイルの名前パターン
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 出力ファイルパスのパターンを生成
    output_pattern = os.path.join(output_dir, filename_pattern)
    # ffmpeg コマンドの実行
    (
        ffmpeg.input(video_path)
        .output(output_pattern, format="image2", vcodec="png")
        .run()
    )


def create_video_from_images(image_dir, output_path):
    # FFmpeg で動画を生成
    ffmpeg.input(os.path.join(image_dir, "%04d.png"), framerate=30).output(
        output_path, vcodec="libx264", pix_fmt="yuv420p", r=60
    ).run()
