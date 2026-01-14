import os
import librosa
import numpy as np
from sklearn.decomposition import PCA

class AudioPipeline:
    def __init__(self, raw_path="data/genres_original"):
        self.raw_path = raw_path
        self.genres = sorted([d for d in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, d))])

    def _extract_stats(self, y, sr):
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harmony, perceptr = librosa.effects.hpss(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(s, ref=np.max)
        
        return {
            'chroma': chroma, 'rms': rms, 'cent': spec_cent, 
            'bw': spec_bw, 'rolloff': rolloff, 'zcr': zcr,
            'harmony': harmony, 'perceptr': perceptr, 'tempo': tempo,
            'mfcc': mfccs, 'mel_db': mel_db
        }

    def run(self, segment_duration=30):
        v1_manual, v2_pca_raw, v3_gtzan, labels = [], [], [], []
        
        for genre in self.genres:
            print(f"Processing: {genre}")
            folder = os.path.join(self.raw_path, genre)
            for file in os.listdir(folder):
                if not file.endswith(".wav"): continue
                path = os.path.join(folder, file)
                
                try:
                    y_full, sr = librosa.load(path, duration=30)
                    samples_per_seg = int(segment_duration * sr)
                    num_segments = 1 if segment_duration == 30 else int(len(y_full) / samples_per_seg)
                    
                    for i in range(num_segments):
                        y_chunk = y_full[i*samples_per_seg : (i+1)*samples_per_seg]
                        raw = self._extract_stats(y_chunk, sr)
                        
                        v1 = np.hstack([np.mean(raw['mfcc'][:13], axis=1), np.mean(raw['cent']), np.mean(raw['zcr'])])
                        v1_manual.append(v1)
                        
                        v2_pca_raw.append(np.mean(raw['mel_db'], axis=1))
                        
                        v3_features = [
                            np.mean(raw['chroma']), np.var(raw['chroma']),
                            np.mean(raw['rms']), np.var(raw['rms']),
                            np.mean(raw['cent']), np.var(raw['cent']),
                            np.mean(raw['bw']), np.var(raw['bw']),
                            np.mean(raw['rolloff']), np.var(raw['rolloff']),
                            np.mean(raw['zcr']), np.var(raw['zcr']),
                            np.mean(raw['harmony']), np.var(raw['harmony']),
                            np.mean(raw['perceptr']), np.var(raw['perceptr']),
                            raw['tempo']
                        ]
                        for m in range(20):
                            v3_features.append(np.mean(raw['mfcc'][m]))
                            v3_features.append(np.var(raw['mfcc'][m]))
                            
                        v3_gtzan.append(v3_features)
                        labels.append(genre)
                except:
                    continue

        X_v1 = np.array(v1_manual)
        X_v2 = PCA(n_components=10).fit_transform(np.array(v2_pca_raw))
        X_v3 = np.array(v3_gtzan)
        y = np.array(labels)
        
        return X_v1, X_v2, X_v3, y