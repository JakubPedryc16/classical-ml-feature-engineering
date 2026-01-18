import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

class GenreClassifier:
    def __init__(self, reports_path="reports"):
        self.reports_path = reports_path
        self.encoder = LabelEncoder()
        os.makedirs(self.reports_path, exist_ok=True)

    def get_groups(self, X):
        segs_per_song = 10 if len(X) > 2000 else 1
        return np.repeat(np.arange(len(X) // segs_per_song + 1), segs_per_song)[:len(X)]

    def get_cv_score(self, model, X, y, groups):
        pipe = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipe, X, y, groups=groups, cv=GroupKFold(5), n_jobs=-1)
        return scores.mean()
    
    def create_tuned_model(self, model_name, params):
        models_map = {
            "SVM": SVC(**params, probability=True, random_state=42),
            "RF": RandomForestClassifier(**params, random_state=42),
            "kNN": KNeighborsClassifier(**params),
            "GradBoost": HistGradientBoostingClassifier(**params, random_state=42)
        }
        return models_map.get(model_name)

    def prepare_data(self, X, y):
        y_encoded = self.encoder.fit_transform(y)
        groups = self.get_groups(X)
        
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test

    def get_base_models(self):
        return [
            ("RF", RandomForestClassifier(random_state=42)),
            ("SVM", SVC(probability=True, random_state=42)),
            ("kNN", KNeighborsClassifier()),
            ("GradBoost", HistGradientBoostingClassifier(random_state=42))
        ]

    def evaluate(self, model, X_test, y_test, name="model"):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.encoder.classes_, 
                    yticklabels=self.encoder.classes_)
        
        plt.title(f'Confusion Matrix: {name}\nAccuracy: {acc:.4f}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.reports_path, f'cm_{name}.png'))
        plt.close()
        return acc