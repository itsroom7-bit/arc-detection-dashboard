"""
실시간 전류 센서 데이터 시뮬레이터
- UYeG-DX 센서의 데이터 형식을 시뮬레이션
- 정상 파형과 아크 파형을 생성
- 1초 윈도우 단위로 데이터 생성 (앨리어싱 문제 해결)
"""

import numpy as np
import time
import threading
from collections import deque
from datetime import datetime

class CurrentSensorSimulator:
    """전류 센서 데이터 시뮬레이터"""
    
    def __init__(self, sampling_rate=35, buffer_size=35):
        """
        Args:
            sampling_rate: 초당 샘플링 횟수 (30~40Hz)
            buffer_size: 1초 윈도우 버퍼 크기
        """
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.interval = 1.0 / sampling_rate
        
        # 데이터 버퍼 (1초간의 데이터 저장)
        self.buffer = deque(maxlen=buffer_size)
        
        # 시뮬레이션 상태
        self.running = False
        self.thread = None
        
        # 현재 모드 (normal, arc_spike, arc_continuous, arc_intermittent)
        self.mode = 'normal'
        self.mode_duration = 0
        self.mode_start_time = time.time()
        
        # 기본 파라미터
        self.base_amplitude = 30  # 기본 전류 진폭 (A)
        self.frequency = 60  # 기본 주파수 (Hz)
        
        # 윈도우 생성용 시간 인덱스
        self.sample_index = 0
        self.current_window = None
        self.window_start_time = None
        
        # 콜백 함수
        self.on_data_callback = None
        self.on_window_callback = None
        
    def set_mode(self, mode, duration=None):
        """
        시뮬레이션 모드 설정
        
        Args:
            mode: 'normal', 'arc_spike', 'arc_continuous', 'arc_intermittent', 'random'
            duration: 모드 지속 시간 (초), None이면 무한
        """
        self.mode = mode
        self.mode_duration = duration
        self.mode_start_time = time.time()
        # 모드 변경 시 새 윈도우 생성
        self._generate_new_window()
        
    def _generate_new_window(self):
        """1초 윈도우 데이터 생성 (앨리어싱 없이)"""
        t = np.linspace(0, 1, self.buffer_size)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # 기본 사인파
        waveform = self.base_amplitude * np.sin(2 * np.pi * self.frequency * t + phase)
        
        # 현재 모드 확인 (random 모드 처리)
        current_mode = self.mode
        if current_mode == 'random':
            elapsed = time.time() - self.mode_start_time
            if elapsed > np.random.uniform(3, 10):
                current_mode = np.random.choice(['normal', 'arc_spike', 'arc_continuous', 'arc_intermittent'], 
                                               p=[0.6, 0.15, 0.15, 0.1])
                self.mode_start_time = time.time()
            else:
                # 이전 모드 유지 또는 기본 정상
                current_mode = 'normal'
        
        # 모드에 따른 변형
        if current_mode == 'normal':
            # 정상 파형: 낮은 노이즈
            noise_level = np.random.uniform(0.01, 0.05)
            waveform += noise_level * self.base_amplitude * np.random.randn(self.buffer_size)
            # 약간의 고조파 (3~5%)
            waveform += np.random.uniform(0.02, 0.05) * self.base_amplitude * np.sin(2 * np.pi * 3 * self.frequency * t + phase)
            
        elif current_mode == 'arc_spike':
            # 스파이크 아크
            n_spikes = np.random.randint(2, 8)
            spike_positions = np.random.choice(self.buffer_size, n_spikes, replace=False)
            for pos in spike_positions:
                spike_magnitude = np.random.uniform(1.5, 4) * self.base_amplitude
                waveform[pos] += np.random.choice([-1, 1]) * spike_magnitude
            # 스파이크 주변 노이즈
            waveform += np.random.uniform(0.1, 0.2) * self.base_amplitude * np.random.randn(self.buffer_size)
                
        elif current_mode == 'arc_continuous':
            # 연속 아크: 고노이즈 + 고조파 왜곡
            noise_level = np.random.uniform(0.2, 0.5)
            waveform += noise_level * self.base_amplitude * np.random.randn(self.buffer_size)
            # 고조파 추가 (8~20%)
            for h in [3, 5, 7, 9]:
                waveform += np.random.uniform(0.08, 0.2) * self.base_amplitude * np.sin(2 * np.pi * h * self.frequency * t + phase)
                
        elif current_mode == 'arc_intermittent':
            # 간헐적 아크
            arc_start = np.random.randint(0, self.buffer_size // 2)
            arc_duration = np.random.randint(self.buffer_size // 4, self.buffer_size // 2)
            arc_end = min(arc_start + arc_duration, self.buffer_size)
            
            waveform[arc_start:arc_end] += np.random.uniform(0.3, 0.8) * self.base_amplitude * np.random.randn(arc_end - arc_start)
            waveform[arc_start:arc_end] *= np.random.uniform(1.1, 1.5)
        
        self.current_window = waveform
        self.sample_index = 0
        self.window_start_time = datetime.now()
        
    def generate_sample(self):
        """단일 샘플 생성 (현재 윈도우에서)"""
        if self.current_window is None or self.sample_index >= self.buffer_size:
            self._generate_new_window()
        
        sample = self.current_window[self.sample_index]
        self.sample_index += 1
        
        return sample
    
    def _simulation_loop(self):
        """시뮬레이션 루프"""
        last_window_time = time.time()
        self._generate_new_window()
        
        while self.running:
            # 샘플 생성
            sample = self.generate_sample()
            timestamp = datetime.now()
            
            # 버퍼에 추가
            self.buffer.append({
                'value': sample,
                'timestamp': timestamp
            })
            
            # 개별 데이터 콜백
            if self.on_data_callback:
                self.on_data_callback(sample, timestamp)
            
            # 1초마다 윈도우 콜백 (윈도우가 완성되었을 때)
            if self.sample_index >= self.buffer_size:
                if self.on_window_callback and len(self.buffer) >= self.buffer_size:
                    window_data = np.array([d['value'] for d in self.buffer])
                    self.on_window_callback(window_data, timestamp)
            
            # 다음 샘플까지 대기
            time.sleep(self.interval)
    
    def start(self):
        """시뮬레이션 시작"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self.thread.start()
            print(f"센서 시뮬레이터 시작 (샘플링 레이트: {self.sampling_rate}Hz)")
    
    def stop(self):
        """시뮬레이션 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            print("센서 시뮬레이터 중지")
    
    def get_current_window(self):
        """현재 1초 윈도우 데이터 반환"""
        if len(self.buffer) >= self.buffer_size:
            return np.array([d['value'] for d in self.buffer])
        return None
    
    def set_amplitude(self, amplitude):
        """기본 진폭 설정"""
        self.base_amplitude = amplitude
        
    def set_frequency(self, frequency):
        """기본 주파수 설정"""
        self.frequency = frequency


class RealTimeDataProcessor:
    """실시간 데이터 처리기"""
    
    def __init__(self, model, sampling_rate=35):
        """
        Args:
            model: ArcDetectionModel 인스턴스
            sampling_rate: 샘플링 레이트
        """
        self.model = model
        self.sampling_rate = sampling_rate
        
        # 결과 저장
        self.predictions = deque(maxlen=60)  # 최근 60초 예측 결과
        self.waveforms = deque(maxlen=60)  # 최근 60초 파형 데이터
        
        # 콜백
        self.on_prediction_callback = None
        
    def process_window(self, window_data, timestamp):
        """
        1초 윈도우 데이터 처리
        
        Args:
            window_data: 1초간의 전류 데이터 배열
            timestamp: 타임스탬프
        """
        # 아크 확률 예측
        arc_probability = self.model.predict_probability(window_data)
        
        # 결과 저장
        result = {
            'timestamp': timestamp,
            'probability': arc_probability,
            'is_arc': arc_probability > 0.5,
            'waveform': window_data.tolist()
        }
        
        self.predictions.append(result)
        self.waveforms.append({
            'timestamp': timestamp,
            'data': window_data.tolist()
        })
        
        # 콜백 호출
        if self.on_prediction_callback:
            self.on_prediction_callback(result)
        
        return result
    
    def get_recent_predictions(self, seconds=60):
        """최근 N초간의 예측 결과 반환"""
        return list(self.predictions)[-seconds:]
    
    def get_statistics(self):
        """통계 정보 반환"""
        if not self.predictions:
            return None
        
        probs = [p['probability'] for p in self.predictions]
        arc_count = sum(1 for p in self.predictions if p['is_arc'])
        
        return {
            'total_windows': len(self.predictions),
            'arc_detected_count': arc_count,
            'arc_ratio': arc_count / len(self.predictions) if self.predictions else 0,
            'avg_probability': np.mean(probs),
            'max_probability': np.max(probs),
            'min_probability': np.min(probs)
        }


# 테스트
if __name__ == "__main__":
    from arc_detection_model import ArcDetectionModel
    
    print("=" * 50)
    print("실시간 데이터 시뮬레이터 테스트")
    print("=" * 50)
    
    # 모델 로드
    model = ArcDetectionModel(sampling_rate=35)
    model.load_model('/home/ubuntu/arc_detection/arc_model.pkl')
    
    # 프로세서 생성
    processor = RealTimeDataProcessor(model, sampling_rate=35)
    
    # 시뮬레이터 생성
    simulator = CurrentSensorSimulator(sampling_rate=35)
    
    # 콜백 설정
    def on_prediction(result):
        status = "🔴 아크 감지!" if result['is_arc'] else "🟢 정상"
        print(f"[{result['timestamp'].strftime('%H:%M:%S')}] {status} - 확률: {result['probability']:.2%}")
    
    processor.on_prediction_callback = on_prediction
    simulator.on_window_callback = lambda data, ts: processor.process_window(data, ts)
    
    # 시뮬레이션 시작
    simulator.start()
    
    # 테스트: 모드 변경
    print("\n[정상 모드로 3초간 실행]")
    simulator.set_mode('normal')
    time.sleep(3.5)
    
    print("\n[스파이크 아크 모드로 3초간 실행]")
    simulator.set_mode('arc_spike')
    time.sleep(3.5)
    
    print("\n[연속 아크 모드로 3초간 실행]")
    simulator.set_mode('arc_continuous')
    time.sleep(3.5)
    
    print("\n[정상 모드로 복귀]")
    simulator.set_mode('normal')
    time.sleep(2.5)
    
    # 통계 출력
    stats = processor.get_statistics()
    print(f"\n통계:")
    print(f"  총 분석 윈도우: {stats['total_windows']}")
    print(f"  아크 감지 횟수: {stats['arc_detected_count']}")
    print(f"  아크 비율: {stats['arc_ratio']:.2%}")
    print(f"  평균 확률: {stats['avg_probability']:.2%}")
    
    simulator.stop()
