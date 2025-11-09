import json, os, subprocess, sys

ANNOT_PATH = "data/mlb-youtube-segmented.json"
SAVE_DIR   = "clips_swing_hit"
YTDLP      = "yt-dlp"

TARGET = {"swing", "hit"}  # 정확히 swing, hit만

os.makedirs(SAVE_DIR, exist_ok=True)

with open(ANNOT_PATH, "r") as f:
    data = json.load(f)

# 엔트리 목록 만들기
if isinstance(data, list):
    entries = data
elif isinstance(data, dict):
    if "clips" in data and isinstance(data["clips"], list):
        entries = data["clips"]
    else:
        entries = [v | {"_id": k} for k, v in data.items()]
else:
    print("[ERROR] JSON 구조 파악 실패:", type(data)); sys.exit(1)

print(f"[INFO] 총 {len(entries)}개 엔트리에서 swing/hit만 필터링...")

def norm_labels(clip):
    for key in ("labels","label","classes"):
        if key in clip:
            labs = clip[key]
            if isinstance(labs, str):
                return {labs.lower().strip()}
            if isinstance(labs, list):
                return {str(x).lower().strip() for x in labs}
    return set()

def get_video_id(clip):
    for k in ("video_id","youtube_id","ytid","id","url"):
        if k in clip: return clip[k]
    return None

def get_times(clip):
    # 데이터셋 스키마에 맞춰 다양한 키 지원
    start = (clip.get("start_time") or clip.get("start") or
             clip.get("time_start") or clip.get("ts") or 0)
    end   = (clip.get("end_time") or clip.get("end") or
             clip.get("time_end") or clip.get("te"))
    return float(start), None if end is None else float(end)

selected = 0
for clip in entries:
    labs = norm_labels(clip)
    if not labs or labs.isdisjoint(TARGET):
        continue  # swing/hit 둘 다 없으면 패스

    vid = get_video_id(clip)
    if not vid: 
        continue
    s, e = get_times(clip)
    if e is None or e <= s:
        continue

    if str(vid).startswith("http"):
        url = vid
        vid_for_name = vid.split("v=")[-1].split("&")[0]
    else:
        url = f"https://www.youtube.com/watch?v={vid}"
        vid_for_name = str(vid)

    out_path = os.path.join(SAVE_DIR, f"{vid_for_name}_{int(s)}_{int(e)}.mp4")
    if os.path.exists(out_path):
        continue

    print(f"▶ {url} [{s:.2f}~{e:.2f}s] 다운로드...")
    subprocess.call([
        sys.executable, "-m", "yt_dlp",  # <-- 이 부분이 핵심입니다.
        "--quiet",
        "--no-warnings",
        "--download-sections", f"*{s}-{e}",
        "-o", out_path,
        url
    ])
    selected += 1

print(f"[완료] swing/hit 클립 저장: {selected}개")
