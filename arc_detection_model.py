"""
아크 감지 AI 모델
- 전류 파형에서 특징을 추출하여 아크 발생 확률을 예측
- 초당 30~40회 샘플링 데이터를 1초 윈도우로 분석
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class ArcFeatureExtractor:
    """전류 파형에서 아크 감지를 위한 특징 추출"""
    
    def __init__(self, sampling_rate=35):
        """
        Args:
            sampling_rate: 초당 샘플링 횟수 (기본 35Hz)
        """
        self.sampling_rate = sampling_rate
        
    def extract_features(self, waveform):
        """
        1초 윈도우의 전류 파형에서 특징 추출
        
        Args:
            waveform: 1초간의 전류 데이터 배열 (30~40 샘플)
            
        Returns:
            특징 벡터 (numpy array)
        """
        features = {}
        
        # 기본 통계적 특징
        features['mean'] = np.mean(waveform)
        features['std'] = np.std(waveform)
        features['max'] = np.max(waveform)
        features['min'] = np.min(waveform)
        features['peak_to_peak'] = features['max'] - features['min']
        features['rms'] = np.sqrt(np.mean(waveform**2))
        
        # 파형 형태 특징
        features['skewness'] = stats.skew(waveform)
        features['kurtosis'] = stats.kurtosis(waveform)
        features['crest_factor'] = features['max'] / features['rms'] if features['rms'] > 0 else 0
        
        # 변화율 특징 (아크 발생 시 급격한 변화)
        diff = np.diff(waveform)
        features['mean_diff'] = np.mean(np.abs(diff))
        features['max_diff'] = np.max(np.abs(diff))
        features['std_diff'] = np.std(diff)
        
        # 제로 크로싱 (영점 교차 횟수)
        zero_crossings = np.where(np.diff(np.signbit(waveform)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(waveform)
        
        # 고조파 분석 (FFT 기반)
        fft_result = fft(waveform)
        fft_magnitude = np.abs(fft_result)[:len(waveform)//2]
        freqs = fftfreq(len(waveform), 1/self.sampling_rate)[:len(waveform)//2]
        
        # 주파수 대역별 에너지
        if len(fft_magnitude) > 0:
            features['fft_mean'] = np.mean(fft_magnitude)
            features['fft_max'] = np.max(fft_magnitude)
            features['fft_std'] = np.std(fft_magnitude)
            
            # 고조파 비율 (기본파 대비 고조파 에너지)
            fundamental_idx = np.argmax(fft_magnitude)
            fundamental_energy = fft_magnitude[fundamental_idx]
            harmonic_energy = np.sum(fft_magnitude) - fundamental_energy
            features['thd'] = harmonic_energy / fundamental_energy if fundamental_energy > 0 else 0
        else:
            features['fft_mean'] = 0
            features['fft_max'] = 0
            features['fft_std'] = 0
            features['thd'] = 0
        
        # 아크 특성 지표
        # 1. 급격한 스파이크 감지
        threshold = features['mean'] + 2 * features['std']
        spikes = np.sum(np.abs(waveform) > threshold)
        features['spike_count'] = spikes
        
        # 2. 불규칙성 지표
        autocorr = np.correlate(waveform - features['mean'], waveform - features['mean'], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        features['autocorr_decay'] = autocorr[1] / autocorr[0] if autocorr[0] > 0 else 0
        
        # 3. 에너지 변동
        window_size = max(len(waveform) // 4, 1)
        energies = [np.sum(waveform[i:i+window_size]**2) for i in range(0, len(waveform)-window_size+1, window_size)]
        features['energy_variance'] = np.var(energies) if len(energies) > 1 else 0
        
        return np.array(list(features.values())), list(features.keys())


class ArcDetectionModel:
    """아크 감지 AI 모델"""
    
    def __init__(self, sampling_rate=35):
        self.feature_extractor = ArcFeatureExtractor(sampling_rate)
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None
        
    def generate_training_data(self, n_samples=1000, window_size=35):
        """
        학습용 합성 데이터 생성
        
        아크 파형 특성:
        - 급격한 전류 스파이크
        - 고주파 노이즈 증가
        - 불규칙한 파형
        - 고조파 왜곡 증가
        """
        X = []
        y = []
        
        for _ in range(n_samples):
            # 시간 배열 생성 (실제 시간 기반)
            t_start = np.random.uniform(0, 100)  # 랜덤 시작 시간
            t = np.linspace(t_start, t_start + 1, window_size)
            
            # 정상 파형 (50% 확률)
            if np.random.random() < 0.5:
                # 정상 사인파 + 약간의 노이즈
                amplitude = np.random.uniform(5, 50)
                frequency = 60  # 60Hz 기본파
                phase = np.random.uniform(0, 2 * np.pi)  # 랜덤 위상
                noise_level = np.random.uniform(0.01, 0.05)  # 낮은 노이즈
                
                waveform = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                waveform += noise_level * amplitude * np.random.randn(window_size)
                
                # 약간의 고조파 추가 (정상 범위: 3~5%)
                waveform += np.random.uniform(0.02, 0.05) * amplitude * np.sin(2 * np.pi * 3 * frequency * t + phase)
                
                y.append(0)  # 정상
            else:
                # 아크 파형 생성
                amplitude = np.random.uniform(5, 50)
                frequency = 60
                phase = np.random.uniform(0, 2 * np.pi)
                
                waveform = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                
                # 아크 특성 추가
                arc_type = np.random.choice(['spike', 'continuous', 'intermittent'])
                
                if arc_type == 'spike':
                    # 급격한 스파이크
                    n_spikes = np.random.randint(2, 8)
                    spike_positions = np.random.choice(window_size, n_spikes, replace=False)
                    for pos in spike_positions:
                        spike_magnitude = np.random.uniform(1.5, 4) * amplitude
                        waveform[pos] += np.random.choice([-1, 1]) * spike_magnitude
                    # 스파이크 주변 노이즈
                    waveform += np.random.uniform(0.1, 0.2) * amplitude * np.random.randn(window_size)
                        
                elif arc_type == 'continuous':
                    # 연속적 아크 (고주파 노이즈)
                    noise_level = np.random.uniform(0.2, 0.5)
                    waveform += noise_level * amplitude * np.random.randn(window_size)
                    # 고조파 왜곡 (높은 THD)
                    for h in [3, 5, 7, 9]:
                        waveform += np.random.uniform(0.08, 0.2) * amplitude * np.sin(2 * np.pi * h * frequency * t + phase)
                        
                else:  # intermittent
                    # 간헐적 아크
                    arc_start = np.random.randint(0, window_size // 2)
                    arc_duration = np.random.randint(window_size // 4, window_size // 2)
                    arc_end = min(arc_start + arc_duration, window_size)
                    
                    waveform[arc_start:arc_end] += np.random.uniform(0.3, 0.8) * amplitude * np.random.randn(arc_end - arc_start)
                    # 급격한 전류 변화
                    waveform[arc_start:arc_end] *= np.random.uniform(1.1, 1.5)
                
                y.append(1)  # 아크
            
            features, self.feature_names = self.feature_extractor.extract_features(waveform)
            X.append(features)
        
        return np.array(X), np.array(y)
    
    def train(self, X=None, y=None):
        """모델 학습"""
        if X is None or y is None:
            print("합성 학습 데이터 생성 중...")
            X, y = self.generate_training_data(n_samples=2000)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 모델 학습
        print("모델 학습 중...")
        self.model.fit(X_train_scaled, y_train)
        
        # 평가
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"학습 정확도: {train_score:.4f}")
        print(f"테스트 정확도: {test_score:.4f}")
        
        self.is_trained = True
        return train_score, test_score
    
    def predict_probability(self, waveform):
        """
        아크 발생 확률 예측
        
        Args:
            waveform: 1초간의 전류 데이터 배열
            
        Returns:
            아크 확률 (0~1)
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        features, _ = self.feature_extractor.extract_features(waveform)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # 확률 예측
        prob = self.model.predict_proba(features_scaled)[0]
        return prob[1]  # 아크 확률 반환
    
    def get_feature_importance(self):
        """특징 중요도 반환"""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def save_model(self, path='arc_model.pkl'):
        """모델 저장"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
        print(f"모델 저장 완료: {path}")
    
    def load_model(self, path='arc_model.pkl'):
        """모델 로드"""
        if os.path.exists(path):
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            print(f"모델 로드 완료: {path}")
            return True
        return False


# 테스트 및 모델 학습
if __name__ == "__main__":
    print("=" * 50)
    print("아크 감지 AI 모델 학습")
    print("=" * 50)
    
    # 모델 생성 및 학습
    model = ArcDetectionModel(sampling_rate=35)
    train_acc, test_acc = model.train()
    
    # 모델 저장
    model.save_model('/home/ubuntu/arc_detection/arc_model.pkl')
    
    # 특징 중요도 출력
    print("\n특징 중요도:")
    importance = model.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_importance[:10]:
        print(f"  {name}: {imp:.4f}")
    
    # 테스트 예측
    print("\n테스트 예측:")
    
    # 정상 파형 테스트
    t = np.linspace(0, 1, 35)
    normal_wave = 30 * np.sin(2 * np.pi * 60 * t) + 0.5 * np.random.randn(35)
    prob = model.predict_probability(normal_wave)
    print(f"  정상 파형 아크 확률: {prob:.2%}")
    
    # 아크 파형 테스트
    arc_wave = 30 * np.sin(2 * np.pi * 60 * t)
    arc_wave += 15 * np.random.randn(35)  # 고노이즈
    arc_wave[10:15] *= 2.5  # 스파이크
    prob = model.predict_probability(arc_wave)
    print(f"  아크 파형 아크 확률: {prob:.2%}")
