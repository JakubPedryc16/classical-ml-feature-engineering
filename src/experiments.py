import os
import pandas as pd
from sklearn.metrics import classification_report
from src.processors import AudioPipeline
from src.data_io import DataIO
from src.models import GenreClassifier
from src.tuner import GenreTuner
from src.visualizer import plot_manual_features, plot_pca_features, plot_rich_features_profile

def run_extraction_and_viz(duration=3):
    print(f"\n=== EKSTRAKCJA CECH (Segmenty: {duration}s) ===")
    io, pipeline = DataIO(), AudioPipeline()
    X1, X2, X3, y = pipeline.run(segment_duration=duration)
    io.save_processed_data(X1, X2, X3, y)
    
    plot_path = io.get_path('plots')
    plot_manual_features(X1, y, save_path=plot_path)
    plot_pca_features(X2, y, save_path=plot_path)
    plot_rich_features_profile(X3, y, save_path=plot_path)

def run_base_evaluation():
    io, pipeline = DataIO(), AudioPipeline()
    data = io.load_processed_data()
    if data[0] is None: return

    X_variants = [data[0], data[1], data[2]]
    X_names = ["V1_Manual", "V2_PCA", "V3_Rich"]
    y = data[3]
    
    clf_mgr = GenreClassifier(reports_path=io.get_path('base'))
    all_results = []

    print("\n=== START KOMPLEKSOWEJ EWALUACJI BAZOWEJ (CV vs TEST) ===")
    
    for i, X in enumerate(X_variants):
        feat_name = X_names[i]
        X_train, X_test, y_train, y_test = clf_mgr.prepare_data(X, y)
        groups = clf_mgr.get_groups(X)

        for model_name, model in clf_mgr.get_base_models():
            avg_cv_acc = clf_mgr.get_cv_score(model, X, y, groups)

            model.fit(X_train, y_train)
            final_test_acc = clf_mgr.evaluate(model, X_test, y_test, f"{model_name}_{feat_name}")
            
            report = classification_report(y_test, model.predict(X_test), 
                                         target_names=clf_mgr.encoder.classes_, output_dict=True)
            io.save_model_results(f"{model_name}_{feat_name}", report, final_test_acc, stage="base")

            all_results.append({
                "Wariant": feat_name, "Model": model_name,
                "CV_Acc": round(avg_cv_acc, 4), "Test_Acc": round(final_test_acc, 4),
                "Diff": round(avg_cv_acc - final_test_acc, 4)
            })
            print(f"[{feat_name}] {model_name:<10} | CV: {avg_cv_acc:.4f} | Test: {final_test_acc:.4f}")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(io.get_path('base'), "full_evaluation_comparison.csv"), index=False)
    print("\n=== FINALNE PODSUMOWANIE (Modele Bazowe) ===\n", df_res.sort_values(by="Test_Acc", ascending=False).to_string(index=False))

def run_full_tuning(variant="V3"):
    io, pipeline = DataIO(), AudioPipeline()
    data = io.load_processed_data()
    
    idx = 2 if variant == "V3" else 0
    X, y = data[idx], data[3]
    
    clf_mgr = GenreClassifier(reports_path=io.get_path('tuning'))
    X_train, X_test, y_train, y_test = clf_mgr.prepare_data(X, y)
    tuner = GenreTuner(X_train, y_train) 

    tuning_results = []
    print(f"\n=== START OPTYMALIZACJI OPTUNA (Wariant: {variant}) ===")

    for model_name in ["SVM", "RF", "kNN", "GradBoost"]:
        best_params, best_cv_score = tuner.optimize(model_name, n_trials=30)
        
        model = clf_mgr.create_tuned_model(model_name, best_params)

        model.fit(X_train, y_train)
        final_test_acc = clf_mgr.evaluate(model, X_test, y_test, f"{model_name}_{variant}_tuned")
        
        report = classification_report(y_test, model.predict(X_test), 
                                     target_names=clf_mgr.encoder.classes_, output_dict=True)
        io.save_model_results(f"{model_name}_{variant}_tuned", report, final_test_acc, stage="tuning")

        tuning_results.append({
            "Model": model_name, "Best_CV_Score": round(best_cv_score, 4),
            "Final_Test_Acc": round(final_test_acc, 4), "Diff": round(best_cv_score - final_test_acc, 4)
        })
        print(f"[{model_name}] Optuna CV: {best_cv_score:.4f} | Final Test: {final_test_acc:.4f}")

    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.to_csv(os.path.join(io.get_path('tuning'), f"tuning_comparison_{variant}.csv"), index=False)
    print(f"\n=== PODSUMOWANIE TUNINGU ===\n", df_tuning.sort_values(by="Final_Test_Acc", ascending=False).to_string(index=False))