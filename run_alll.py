import os
import subprocess

VIDEO_DIR = "data/cropped_videos"

for fname in os.listdir(VIDEO_DIR):
    if fname.endswith(".mp4"):
        path = os.path.join(VIDEO_DIR, fname)
        print(f"\n=== Processing {path} ===\n")
        subprocess.run(["python", "main_pipeline.py", "--input", path])
