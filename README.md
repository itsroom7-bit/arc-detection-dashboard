# 실시간 아크 감지 모니터링 시스템

스마트 전류 센서 기반 AI 아크 파형 분석 시스템

## 개요

이 시스템은 UYeG-DX 스마트 전류 센서에서 초당 30~40회 샘플링된 데이터를 받아 실시간으로 아크 파형을 감지하고 확률을 예측하는 AI 기반 모니터링 시스템입니다.

### 주요 기능

- **실시간 아크 감지**: 1초 윈도우 단위로 전류 파형을 분석하여 아크 발생 확률 예측
- **AI 기반 분류**: Gradient Boosting 알고리즘을 사용한 고정확도 아크 파형 분류
- **실시간 시각화**: 웹 기반 대시보드로 파형 및 확률 히스토리 실시간 모니터링
- **다양한 아크 유형 감지**: 스파이크, 연속, 간헐적 아크 패턴 인식

## 시스템 구성

```
arc_detection/
├── arc_detection_model.py   # AI 모델 (특징 추출 + 분류)
├── data_simulator.py        # 센서 데이터 시뮬레이터
├── app.py                   # Flask 웹 서버
├── arc_model.pkl            # 학습된 모델 파일
├── templates/
│   └── index.html           # 웹 대시보드 UI
└── README.md
```

## 기술 스택

- **AI/ML**: scikit-learn (Gradient Boosting Classifier)
- **신호 처리**: NumPy, SciPy (FFT, 통계 분석)
- **웹 서버**: Flask
- **프론트엔드**: HTML5, CSS3, JavaScript, Chart.js

## 아크 감지 알고리즘

### 특징 추출 (Feature Extraction)

1초 윈도우의 전류 파형에서 다음 특징들을 추출합니다:

| 특징 | 설명 |
|------|------|
| **THD (Total Harmonic Distortion)** | 총 고조파 왜곡률 - 가장 중요한 특징 |
| **RMS** | 실효값 |
| **Peak-to-Peak** | 최대-최소 진폭 차이 |
| **Crest Factor** | 파고율 |
| **Skewness/Kurtosis** | 파형 비대칭성/첨도 |
| **Zero Crossing Rate** | 영점 교차율 |
| **Spike Count** | 스파이크 발생 횟수 |
| **Energy Variance** | 에너지 변동성 |

### 아크 파형 특성

- **스파이크 아크**: 급격한 전류 스파이크 발생
- **연속 아크**: 고노이즈 + 고조파 왜곡 증가
- **간헐적 아크**: 주기적인 아크 발생 패턴

## 설치 및 실행

### 요구사항

```bash
pip install numpy scipy scikit-learn flask joblib
```

### 모델 학습

```bash
python arc_detection_model.py
```

### 서버 실행

```bash
python app.py
```

웹 브라우저에서 `http://localhost:5000` 접속

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 메인 대시보드 |
| `/api/start` | POST | 모니터링 시작 |
| `/api/stop` | POST | 모니터링 중지 |
| `/api/mode` | POST | 시뮬레이션 모드 변경 |
| `/api/data` | GET | 최신 데이터 조회 |
| `/api/history` | GET | 히스토리 데이터 조회 |
| `/api/statistics` | GET | 통계 정보 조회 |

## 시뮬레이션 모드

- **normal**: 정상 전류 파형
- **arc_spike**: 스파이크 아크 발생
- **arc_continuous**: 연속 아크 발생
- **arc_intermittent**: 간헐적 아크 발생
- **random**: 자동 모드 전환

## 실제 센서 연동

실제 UYeG-DX 센서와 연동하려면 `data_simulator.py`의 `CurrentSensorSimulator` 클래스를 수정하여 Modbus TCP/IP 프로토콜로 센서 데이터를 수신하도록 구현합니다.

```python
# 예시: Modbus TCP 연동
from pymodbus.client import ModbusTcpClient

class RealSensorReader:
    def __init__(self, host='192.168.1.100', port=502):
        self.client = ModbusTcpClient(host, port=port)
        
    def read_current(self):
        # UYeG-DX 레지스터 주소에서 전류 데이터 읽기
        result = self.client.read_holding_registers(address=0x0000, count=3)
        return result.registers
```

## 모델 성능

- **학습 정확도**: 100%
- **테스트 정확도**: 99.75%
- **주요 특징**: THD (98.84% 중요도)

## 라이선스

MIT License

## 참고 자료

- UYeG-DX 스마트 EOCR 카탈로그
- IEC 60255 전류 보호 표준
- IEEE 불평형 규격
