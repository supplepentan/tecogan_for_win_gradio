from pathlib import Path


class ConfigBase:
    INPUT_DIRECTORY = "input"
    INPUT_MOVIE_FILE = "input_movie.mp4"
    OUTPUT_DIRECTORY = "output"
    IMAGES_DIRECGORY = "images"
    PRETRAINED_MODELS_DIRECTORY = "pretrained_models"


class ConfigCmd(ConfigBase):
    PRETRAINED_MODEL = "TecoGAN_4x_BD_REDS_iter500K.pth"


class ConfigWeb(ConfigBase):
    pass


"""
cmd:
  input:
    directory_name: "input_dir"
    images_directory_name: "images"
    movie_file_name: "input_movie.mp4"
  output:
    directory_name: "output_dir"
    movie_file_name: "output_movie.mp4"
    audio_file_name: "output_audio.wav"
  pretrained_models_directory: "pretrained_models/TecoGAN_4x_BD_REDS_iter500K.pth"

web:
  input:
    directory_name: "input_dir"
    images_directory_name: "images"
    movie_file_name: "input_movie.mp4"
  output:
    directory_name: "output_dir"
    movie_file_name: "output_movie.mp4"
    audio_file_name: "output_audio.wav"

pretrained_models_directory: "pretrained_models"
"""
