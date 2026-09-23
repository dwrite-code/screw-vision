import cv2
from ultralytics import YOLO

# YOLO 기본 모델 불러오기
model = YOLO("yolo11n.pt")

# 카메라 연결
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("✅ AI 카메라 시작!")
print("종료하려면 Q를 누르세요.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # AI 분석
    results = model(frame, verbose=False)

    # AI가 인식한 결과를 화면에 표시
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow(
        "AI Camera Test",
        annotated_frame
    )

    # Q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()