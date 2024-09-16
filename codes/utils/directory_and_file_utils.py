from pathlib import Path
import shutil


def remove_path(target: str) -> None:
    """指定されたパスを削除します。ディレクトリの場合、再帰的に削除します。"""
    target_path = Path(target)
    if target_path.exists():
        shutil.rmtree(target_path) if target_path.is_dir() else target_path.unlink()


def clean_input_directory(input_directory) -> None:
    remove_path(input_directory)
    return
