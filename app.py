"""
실시간 아크 감지 시각화 웹 애플리케이션
- Flask + REST API를 사용한 실시간 데이터 스트리밍
- 1초마다 아크 확률 예측 및 파형 시각화
"""

from flask import Flask, render_template, jsonify, request
import threading
import json
import time
import os
from datetime import datetime
from collections import deque
import numpy as np

from arc_detection_model import ArcDetectionModel
from data_simulator import CurrentSensorSimulator, RealTimeDataProcessor

app = Flask(__name__)

# 전역 변수
model = None
simulator = None
processor = None
is_running = False
initialized = False

# 실시간 데이터 저장
latest_data = {
    'timestamp': None,
    'probability': 0,
    'is_arc': False,
    'waveform': [],
    'status': 'stopped'
}

# 히스토리 데이터 (최근 60초)
history_data = deque(maxlen=60)

def get_model_path():
    """모델 파일 경로 반환 (상대 경로 사용)"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'arc_model.pkl')

def initialize():
    """모델 및 시뮬레이터 초기화"""
    global model, simulator, processor, initialized
    
    if initialized:
        return
    
    print("시스템 초기화 중...")
    
    # 모델 로드
    model = ArcDetectionModel(sampling_rate=35)
    model_path = get_model_path()
    
    if os.path.exists(model_path):
        print(f"모델 로드 중: {model_path}")
        model.load_model(model_path)
    else:
        print("모델 파일이 없습니다. 새로 학습합니다...")
        model.train()
        model.save_model(model_path)
    
    # 프로세서 생성
    processor = RealTimeDataProcessor(model, sampling_rate=35)
    
    # 시뮬레이터 생성
    simulator = CurrentSensorSimulator(sampling_rate=35)
    
    # 콜백 설정
    def on_prediction(result):
        global latest_data
        latest_data = {
            'timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'probability': float(result['probability']),
            'is_arc': bool(result['is_arc']),
            'waveform': result['waveform'],
            'status': 'running'
        }
        history_data.append({
            'timestamp': result['timestamp'].strftime('%H:%M:%S'),
            'probability': float(result['probability'])
        })
    
    processor.on_prediction_callback = on_prediction
    simulator.on_window_callback = lambda data, ts: processor.process_window(data, ts)
    
    initialized = True
    print("시스템 초기화 완료!")

# 앱 시작 시 초기화 (Gunicorn 호환)
with app.app_context():
    initialize()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/api/start', methods=['POST'])
def start_monitoring():
    """모니터링 시작"""
    global is_running, latest_data
    
    # 초기화 확인
    if not initialized:
        initialize()
    
    if not is_running:
        mode = request.json.get('mode', 'random') if request.json else 'random'
        simulator.set_mode(mode)
        simulator.start()
        is_running = True
        latest_data['status'] = 'running'
        return jsonify({'status': 'started', 'mode': mode})
    return jsonify({'status': 'already_running'})

@app.route('/api/stop', methods=['POST'])
def stop_monitoring():
    """모니터링 중지"""
    global is_running, latest_data
    
    if is_running:
        simulator.stop()
        is_running = False
        latest_data['status'] = 'stopped'
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'already_stopped'})

@app.route('/api/mode', methods=['POST'])
def set_mode():
    """시뮬레이션 모드 변경"""
    mode = request.json.get('mode', 'normal')
    if is_running:
        simulator.set_mode(mode)
        return jsonify({'status': 'mode_changed', 'mode': mode})
    return jsonify({'status': 'not_running'})

@app.route('/api/data')
def get_data():
    """최신 데이터 조회"""
    return jsonify(latest_data)

@app.route('/api/history')
def get_history():
    """히스토리 데이터 조회"""
    return jsonify(list(history_data))

@app.route('/api/statistics')
def get_statistics():
    """통계 정보 조회"""
    if processor:
        stats = processor.get_statistics()
        if stats:
            return jsonify(stats)
    return jsonify({})

@app.route('/api/status')
def get_status():
    """시스템 상태 조회"""
    return jsonify({
        'is_running': is_running,
        'mode': simulator.mode if simulator else None,
        'sampling_rate': simulator.sampling_rate if simulator else None,
        'initialized': initialized
    })

if __name__ == '__main__':
    print("=" * 50)
    print("아크 감지 실시간 모니터링 시스템")
    print("=" * 50)
    
    # 서버 시작
    print("\n웹 서버 시작: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
