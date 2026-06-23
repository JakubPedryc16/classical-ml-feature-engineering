import os
import numpy as np
import pandas as pd

class DataIO:
    def __init__(self, processed_path="data/processed", reports_path="reports"):
        self.processed_path = processed_path
        self.reports_path = reports_path
        
        self.subdirs = {
            'plots': os.path.join(reports_path, "plots"),
            'base': os.path.join(reports_path, "base_eval"),
            'tuning': os.path.join(reports_path, "tuning")
        }
        
        os.makedirs(self.processed_path, exist_ok=True)
        for path in self.subdirs.values():
            os.makedirs(path, exist_ok=True)

    def save_processed_data(self, X_v1, X_v2, X_v3, y):
        np.save(os.path.join(self.processed_path, 'X_v1.npy'), X_v1)
        np.save(os.path.join(self.processed_path, 'X_v2.npy'), X_v2)
        np.save(os.path.join(self.processed_path, 'X_v3.npy'), X_v3)
        np.save(os.path.join(self.processed_path, 'y_labels.npy'), y)
        
        self._save_csv(X_v1, y, 'features_v1_manual.csv', mode='v1')
        self._save_csv(X_v2, y, 'features_v2_pca.csv', mode='v2')
        self._save_csv(X_v3, y, 'features_v3_rich.csv', mode='v3')

    def _save_csv(self, X, y, filename, mode='v1'):
        if mode == 'v2':
            columns = [f'PC_{i+1}' for i in range(X.shape[1])]
        elif mode == 'v1':
            mfcc_cols = [f'MFCC_{i+1}' for i in range(13)]
            columns = mfcc_cols + ['Spectral_Centroid', 'Zero_Crossing_Rate']
        else:
            base_cols = [
                'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
                'spectral_centroid_mean', 'spectral_centroid_var',
                'spectral_bandwidth_mean', 'spectral_bandwidth_var',
                'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
                'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo'
            ]
            mfcc_cols = []
            for i in range(1, 21):
                mfcc_cols.extend([f'mfcc{i}_mean', f'mfcc{i}_var'])
            columns = base_cols + mfcc_cols

        df = pd.DataFrame(X, columns=columns)
        df['label'] = y
        df.to_csv(os.path.join(self.processed_path, filename), index=False)

    def load_processed_data(self):
        try:
            path = self.processed_path
            return (np.load(os.path.join(path, 'X_v1.npy'), allow_pickle=True),
                    np.load(os.path.join(path, 'X_v2.npy'), allow_pickle=True),
                    np.load(os.path.join(path, 'X_v3.npy'), allow_pickle=True),
                    np.load(os.path.join(path, 'y_labels.npy'), allow_pickle=True))
        except FileNotFoundError:
            return None, None, None, None

    def save_model_results(self, model_name, report_dict, accuracy, stage="base"):
        target_dir = self.subdirs.get(stage, self.reports_path)
        file_path = os.path.join(target_dir, f"metrics_{model_name}.txt")
        
        df = pd.DataFrame(report_dict).transpose()
        cols_to_format = ['precision', 'recall', 'f1-score']
        for col in cols_to_format:
            df[col] = df[col].apply(lambda x: f"{x*100:6.2f}%" if isinstance(x, (int, float)) else x)
        df['support'] = df['support'].apply(lambda x: f"{int(x):>7}" if isinstance(x, (int, float)) else x)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"=== RAPORT KLASYFIKACJI ({stage.upper()}): {model_name} ===\n")
            f.write(f"OGÓLNY WYNIK (Accuracy): {accuracy*100:.2f}%\n\n")
            f.write(f"{'Gatunek':<15} | {'Prec.':>8} | {'Rec.':>8} | {'F1':>8} | {'Próbek':>8}\n")
            f.write("-" * 55 + "\n")
            for index, row in df.iterrows():
                if index in ['accuracy', 'macro avg', 'weighted avg']: continue
                f.write(f"{index:<15} | {row['precision']} | {row['recall']} | {row['f1-score']} | {row['support']}\n")
            f.write("-" * 55 + "\n")
            f.write(f"Macro Average   | {df.loc['macro avg', 'precision']} | {df.loc['macro avg', 'recall']} | {df.loc['macro avg', 'f1-score']} |\n")

    def save_summary_table(self, results_list, filename="base_evaluation_results.csv"):
        df_res = pd.DataFrame(results_list)
        summary_table = df_res.pivot(index="Model", columns="Wariant", values="CV_Accuracy")
        summary_table.to_csv(os.path.join(self.subdirs['base'], filename))
        return summary_table

    def get_path(self, key):
        return self.subdirs.get(key, self.reports_path)