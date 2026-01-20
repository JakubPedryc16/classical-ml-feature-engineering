import optuna
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

class GenreTuner:
    def __init__(self, X, y):
        self.encoder = LabelEncoder()
        self.y = self.encoder.fit_transform(y)
        self.X = X
        
        segs_per_song = 10 if len(self.X) > 2000 else 1
        self.groups = np.repeat(np.arange(len(self.X) // segs_per_song + 1), segs_per_song)[:len(self.X)]

    def objective(self, trial, model_name):
        if model_name == "SVM":
            c = trial.suggest_float('C', 1e-3, 1e2, log=True)
            gamma = trial.suggest_float('gamma', 1e-3, 1e1, log=True)
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            clf = SVC(C=c, gamma=gamma, kernel=kernel, random_state=42)

        elif model_name == "RF":
            n_estimators = trial.suggest_int('n_estimators', 100, 400)
            max_depth = trial.suggest_int('max_depth', 10, 40)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                        min_samples_split=min_samples_split, random_state=42, n_jobs=-1)

        elif model_name == "kNN":
            n_neighbors = trial.suggest_int('n_neighbors', 3, 25)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        elif model_name == "GradBoost":
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            max_iter = trial.suggest_int('max_iter', 100, 400)
            max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 31, 127)
            clf = HistGradientBoostingClassifier(learning_rate=learning_rate, max_iter=max_iter, 
                                                 max_leaf_nodes=max_leaf_nodes, random_state=42)

        pipeline = make_pipeline(StandardScaler(), clf)
        gkf = GroupKFold(n_splits=5)
        score = cross_val_score(pipeline, self.X, self.y, groups=self.groups, cv=gkf, n_jobs=-1)
        
        return score.mean()

    def optimize(self, model_name, n_trials=30):
        print(f"\n--- Optymalizacja {model_name} (Pipeline + GroupKFold) ---")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, model_name), n_trials=n_trials)
        
        print(f"Najlepszy wynik CV dla {model_name}: {study.best_value:.4f}")
        return study.best_params, study.best_value