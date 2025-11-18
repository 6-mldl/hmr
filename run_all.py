import os
import subprocess

VIDEO_DIR = "data/cropped_videos"
OUTPUT_ROOT = "output_all"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

for fname in sorted(os.listdir(VIDEO_DIR)):
    if fname.endswith(".mp4"):
        input_path = os.path.join(VIDEO_DIR, fname)

        # 영상 하나당 고유한 output 폴더
        video_name = fname.replace(".mp4", "")
        output_dir = os.path.join(OUTPUT_ROOT, video_name)

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== Processing {input_path} ===")
        print(f"Output → {output_dir}\n")

        subprocess.run([
            "python", "main_pipeline.py",
            "--input", input_path,
            "--output_dir", output_dir
        ])
