import cv2
from ultralytics import YOLO
import torch 

# --- 1. 모델 로드 및 GPU 설정 ---
DEVICE = 'cuda:0' 

CONFIDENCE_THRESHOLD = 70.0 

try:
    # 학습된 best.pt만 사용합니다.
    model = YOLO('best.pt', task='classify').to(DEVICE)
    CLASSIFIER_NAMES = model.names
    print("모델 GPU 로드 성공: 최고 정확도 모드.")
except Exception as e:
    model = YOLO('best.pt', task='classify')
    CLASSIFIER_NAMES = model.names
    print("GPU 로드 실패! CPU로 실행합니다.")


# --- 2. 카메라 설정 ---
cap = cv2.VideoCapture(0) 

# --- 함수: ROI 분석 및 GUI 그리기 (중복 코드 제거) ---
def analyze_and_draw_roi(frame, x_start_ratio, x_end_ratio, y_offset, display_offset, W, H):
    """
    단일 ROI를 분석하고 결과 텍스트와 박스를 프레임에 그리는 함수
    :param x_start_ratio: ROI 시작 X 좌표 비율 (0.0 ~ 1.0)
    :param x_end_ratio: ROI 끝 X 좌표 비율 (0.0 ~ 1.0)
    :param y_offset: ROI 세로 여백 비율 (예: 0.05)
    :param display_offset: GUI 텍스트 출력 X 좌표 보정값
    """
    
    # 픽셀 좌표 계산
    x_start = int(W * x_start_ratio)
    x_end = int(W * x_end_ratio)
    y_start = int(H * y_offset)
    y_end = int(H * (1 - y_offset))
    
    # ROI Crop
    roi_crop = frame[y_start:y_end, x_start:x_end]
    
    # Crop된 영역이 유효한지 확인
    if roi_crop.shape[0] == 0 or roi_crop.shape[1] == 0:
        return frame # 유효하지 않으면 원본 프레임 반환

    # GPU 추론 실행
    results = model.predict(source=roi_crop, device=DEVICE, verbose=False) 
    
    # 결과 추출
    probs = results[0].probs
    predicted_name = CLASSIFIER_NAMES[probs.top1]
    confidence = probs.top5conf[0].item() * 100 
    
    # 최종 상태 결정
    if confidence < CONFIDENCE_THRESHOLD:
        final_status = "No Plant Detected"
        text_color = (150, 150, 150) # 회색
    else:
        final_status = f"Status: {predicted_name} ({confidence:.1f}%)"
        text_color = (0, 255, 0) if predicted_name == 'Healthy' else (0, 0, 255) # 초록/빨강

    # GUI 표시
    # 텍스트 표시 (상단)
    cv2.putText(frame, final_status, (display_offset + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
    
    # ROI 영역 표시 (노란색 네모 박스)
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)
    cv2.putText(frame, "ROI Target", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    
    return frame

# --- 3. 실시간 추론 및 GUI 루프 ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    H, W, _ = frame.shape 
    
    # --- ROI 설정 ---
    
    # 🟢 Y축 여백 설정: 0.2(20%)에서 0.05(5%)로 줄여 ROI를 세로로 확장합니다.
    Y_MARGIN = 0.20 
    
    # 1. 좌측 ROI (전체 폭의 0% ~ 50%, Y축은 5%~95% 사용)
    frame = analyze_and_draw_roi(frame, 0.0, 0.5, Y_MARGIN, 0, W, H)
    
    # 2. 우측 ROI (전체 폭의 50% ~ 100%, Y축은 5%~95% 사용)
    frame = analyze_and_draw_roi(frame, 0.5, 1.0, Y_MARGIN, int(W * 0.5), W, H)
    
    # 구분선 (중앙에 세로선 추가)
    cv2.line(frame, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)

    # 창 표시 및 종료
    cv2.imshow("Realtime Dual Plant Classifier (GPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# --- 4. 종료 처리 ---
cap.release()
cv2.destroyAllWindows()