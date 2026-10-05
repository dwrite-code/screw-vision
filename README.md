# 나사 인식 (YOLO 객체 검출)

카메라로 나사·볼트 종류를 인식한다.

## 쓰는 순서

```bash
python list_cameras.py                  # 카메라 번호 확인 (아이패드 쓸 때)
python train_detect.py --epochs 3       # 짧게 돌려 동작 확인
python train_detect.py                  # 본 학습 (50에폭)
python detect_camera.py                 # 카메라로 인식
```

## 파일

| 파일 | 역할 |
|---|---|
| `config.py` | 경로·에폭·카메라 번호 등 설정 |
| `train_detect.py` | YOLO 학습 |
| `detect_camera.py` | 카메라 + 학습한 모델로 인식 |
| `list_cameras.py` | 연결된 카메라 번호 찾기 |
| `fastener/` | 데이터셋 (1,901장 / 7종) |

## 인식 가능한 종류

combo-screw, flat-screw, hex-bolt, hex-screw, phillips-screw, square-screw, undefined

규격(M4x10 vs M4x12)은 구분하지 못한다. 공개 데이터셋에 그 라벨이 없기 때문이다.
규격 구분은 직접 찍은 사진으로 따로 학습해야 한다 (`classify/` 참고).
