import cv2
import numpy as np
from ultralytics import YOLO
import os

# 1. 로컬 PC에 있는 원본 비디오 폴더 경로
LOCAL_INPUT_FOLDER = "original_videos"
# 2. 크롭된 비디오를 저장할 폴더 경로
LOCAL_OUTPUT_FOLDER = "cropped_videos"
# 3. 로컬 PC에 있는 YOLO 모델 파일의 '전체' 경로
LOCAL_MODEL_PATH = "pitcher_hitter_catcher_detector_v3.pt" # 예: "C:/models/yolo_v3.pt"
# -----------------------------------------------------------------


### 2. 로컬 환경 준비 ###
# 결과물 폴더 생성
os.makedirs(LOCAL_OUTPUT_FOLDER, exist_ok=True)

# 2-1. (중요) 로컬 모델 파일 확인
print(f"로컬 모델 파일 확인 중... ({LOCAL_MODEL_PATH})")
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Error: 모델 파일을 찾을 수 없습니다!")
    print(f"지정한 경로에 '{LOCAL_MODEL_PATH}' 파일이 있는지 확인하세요.")
    raise FileNotFoundError(LOCAL_MODEL_PATH)
else:
    print("모델 파일 확인 완료.")

# 2-2. 로컬에 있는 모델 로드
print(f"모델 로드 중... ({LOCAL_MODEL_PATH})")
model = YOLO(LOCAL_MODEL_PATH)

# 2-3. 'hitter' 클래스 ID 자동 찾기
hitter_class_id = None
target_class_name = "hitter"
for class_id, class_name in model.names.items():
    if class_name.lower() == target_class_name.lower():
        hitter_class_id = class_id
        break
if hitter_class_id is None:
    print(f"Error: '{target_class_name}' 클래스를 찾을 수 없습니다.")
    raise Exception("Class ID Not Found")
else:
    print(f"'{target_class_name}' 클래스 ID를 찾았습니다: {hitter_class_id}")


### 3. 비디오 파일 목록 가져오기 ###
print(f"'{LOCAL_INPUT_FOLDER}' 폴더에서 비디오 파일 검색합니다...")
try:
    all_files_list = os.listdir(LOCAL_INPUT_FOLDER)
except FileNotFoundError:
    print(f"Error: 입력 폴더를 찾을 수 없습니다! 경로를 확인하세요: {LOCAL_INPUT_FOLDER}")
    raise

if not all_files_list:
    print(f"Error: '{LOCAL_INPUT_FOLDER}' 폴더가 비어있습니다.")
    raise Exception("Input Folder Empty")

print(f"총 {len(all_files_list)}개의 항목(파일/폴더)을 찾았습니다.")
print("==============================================")


### 4. (루프 시작) 각 비디오를 하나씩 처리 ###
for filename in all_files_list:

    # 4-1. 처리할 파일의 전체 경로 정의
    input_video_path = os.path.join(LOCAL_INPUT_FOLDER, filename)

    # 4-2. 출력 파일 경로 설정 (출력은 .mp4로 통일)
    output_filename = f"{os.path.splitext(filename)[0]}_cropped.mp4"
    output_video_path = os.path.join(LOCAL_OUTPUT_FOLDER, output_filename)

    # 4-3. 이미 처리된 파일은 건너뛰기
    if os.path.exists(output_video_path):
        print(f"\n[{filename}] (은)는 이미 처리되었습니다. 건너뜁니다.")
        continue

    # 4-4. (안전장치) 폴더(디렉터리)는 건너뛰기
    if not os.path.isfile(input_video_path):
        print(f"\n[{filename}] (은)는 파일이 아닙니다. 건너뜁니다.")
        continue

    # 4-5. "읽기 -> 처리 -> 저장" (로컬이라 단순함)
    try:
        # (1) 처리: 로컬에서 비디오 열기
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: [{filename}] 비디오 파일을 열 수 없습니다. 건너뜁니다.")
            continue

        print(f"[{filename}] 비디오 파일 확인. 처리를 시작합니다...")

        # 로컬에 저장할 결과물 비디오 라이터 설정
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (416, 416))

        # "스위치" 역할: Hitter가 한 번이라도 감지되었는지 여부
        hitter_detected_at_least_once = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # (중요) 매 프레임마다 predict 실행
            # 이 부분이 GPU를 사용합니다.
            results = model.predict(frame, verbose=False, classes=[hitter_class_id], imgsz=416)
            hitter_crop = None

            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                hitter_crop = frame[y1:y2, x1:x2]
                if hitter_crop.size == 0:
                    hitter_crop = None
                    continue
                break

            # --- 감지 성공 시에만 비디오에 씀 ---
            if hitter_crop is not None:
                # (성공) "스위치 ON"
                hitter_detected_at_least_once = True 

                canvas = np.zeros((416, 416, 3), dtype=np.uint8)
                crop_h, crop_w = hitter_crop.shape[:2]
                final_crop_to_paste = hitter_crop

                if crop_h > 416 or crop_w > 416:
                    scale = min(416 / crop_w, 416 / crop_h)
                    new_w = int(crop_w * scale)
                    new_h = int(crop_h * scale)
                    if new_w > 0 and new_h > 0:
                        final_crop_to_paste = cv2.resize(hitter_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        crop_h, crop_w = new_h, new_w
                    else:
                        final_crop_to_paste = None

                if final_crop_to_paste is not None:
                    x_offset = (416 - crop_w) // 2
                    y_offset = (416 - crop_h) // 2
                    canvas[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = final_crop_to_paste

                    # 감지에 성공한 프레임(canvas)만 비디오에 쓴다
                    out.write(canvas)
            
            # (수정) 'else' (감지 실패) 시에는 아무것도 하지 않음 (프레임 건너뜀)

        cap.release()
        out.release()
        print(f"[{filename}] 로컬 처리 완료.")

        # (2) 저장: Hitter가 아예 안 나온 영상(빈 파일)은 삭제
        if not hitter_detected_at_least_once:
                print(f"Warning: [{filename}] 영상에서 'hitter'를 한 번도 감지하지 못했습니다. 빈 파일을 삭제합니다.")
                if os.path.exists(output_video_path):
                     os.remove(output_video_path) # 로컬에 생성된 빈 파일 삭제
        else:
             print(f"✅ [{filename}] 처리 완료! -> '{output_video_path}'에 저장됨")

    except Exception as e:
        print(f"Error: [{filename}] 처리 중 심각한 오류 발생: {e}")


### 5. (루프 종료) 모든 작업 완료 ###
cv2.destroyAllWindows()
print("\n==============================================")
print(f"모든 비디오 처리가 완료되었습니다.")
print(f"결과물은 '{LOCAL_OUTPUT_FOLDER}' 폴더에 저장되었습니다.")