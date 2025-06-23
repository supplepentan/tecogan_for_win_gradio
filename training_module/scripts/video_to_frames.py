import argparse
import os
import sys
import cv2
import glob


def video_to_frames(video_path, output_dir):
    """
    Converts a video file into a sequence of image frames.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output frames.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Save frame as a PNG file with 8-digit zero-padding
        frame_filename = os.path.join(output_dir, f"{frame_count:08d}.png")
        cv2.imwrite(frame_filename, frame)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...", end="\r")

    cap.release()
    print(
        f"\nFinished. Extracted {frame_count} frames from {os.path.basename(video_path)}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert video files into sequences of image frames for TecoGAN training."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to a single input video file.")
    group.add_argument(
        "--video_dir",
        type=str,
        help="Path to a directory containing input video files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save the output sequence folders (e.g., '.../MyCustomDataset/Raw').",
    )
    args = parser.parse_args()

    # --- Get a list of video files to process ---
    if args.video_dir:
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.video_dir, ext)))
        if not video_files:
            print(f"Error: No video files found in {args.video_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(video_files)} video(s) to process.")
    else:
        video_files = [args.video]

    # --- Process each video file ---
    for i, video_path in enumerate(sorted(video_files)):
        # Each video will be placed in a numbered sequence folder (001, 002, etc.)
        sequence_folder_name = f"{i + 1:03d}"
        output_sequence_dir = os.path.join(args.out_dir, sequence_folder_name)

        print("\n" + "=" * 50)
        print(
            f"Processing video {i + 1}/{len(video_files)}: {os.path.basename(video_path)}"
        )
        print(f"Outputting frames to: {output_sequence_dir}")
        print("=" * 50)

        video_to_frames(video_path, output_sequence_dir)

    print("\nAll videos have been processed.")
