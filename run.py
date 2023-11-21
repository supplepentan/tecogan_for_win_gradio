import os
import shutil

from codes.utils import base_utils, movie_utils, inference_utils

INPUT_DIR = "data"
INPUT_IMAGES_FOLDER = "input"
INPUT_MOVIE_FILENAME = "input.mp4"

OUTPUT_DIR = "results"
OUTPUT_IMAGES_FOLDER = "output_images"
OUTPUT_MOVIE_FILENAME = "output.mp4"


def main():
    movie_utils.extract_images_from_video(
        os.path.join(INPUT_DIR, INPUT_MOVIE_FILENAME),
        os.path.join(INPUT_DIR, INPUT_IMAGES_FOLDER),
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


if __name__ == "__main__":
    main()
