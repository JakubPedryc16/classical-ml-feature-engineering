import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from src.data_io import DataIO

def plot_manual_features(X, y, save_path=None):
    plt.figure(figsize=(12, 8))
    genres = np.unique(y)
    for genre in genres:
        mask = (y == genre)
        plt.scatter(X[mask, 13], X[mask, 14], label=genre, alpha=0.6, edgecolors='w', s=40)
    
    plt.xlabel('Spectral Centroid (Brightness)')
    plt.ylabel('Zero Crossing Rate (Noisiness)')
    plt.title(f'V1: Manual Features Analysis\nSamples: {len(y)}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Genres")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'v1_manual_features.png'), dpi=300)
    else:
        plt.show()
    plt.close()

def plot_pca_features(X_pca, y, save_path=None):
    plt.figure(figsize=(12, 8))
    genres = np.unique(y)
    for genre in genres:
        mask = (y == genre)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=genre, alpha=0.6, edgecolors='w', s=40)
    
    plt.xlabel('PC 1 (Main variance)')
    plt.ylabel('PC 2 (Secondary variance)')
    plt.title('V2: PCA Feature Space (Mel-Spectrograms)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Genres")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'v2_pca_features.png'), dpi=300)
    else:
        plt.show()
    plt.close()

def plot_rich_features_profile(X, y, save_path=None):
    plt.figure(figsize=(14, 7))
    genres = np.unique(y)
    
    mfcc_mean_indices = [17 + i*2 for i in range(20)]
    mfcc_part = X[:, mfcc_mean_indices]
    
    for genre in genres:
        genre_mean = np.mean(mfcc_part[y == genre], axis=0)
        plt.plot(range(1, 21), genre_mean, label=genre, marker='o', alpha=0.8)
    
    plt.xticks(range(1, 21))
    plt.xlabel('MFCC Coefficient Index')
    plt.ylabel('Mean Value')
    plt.title('V3: Rich Features - MFCC Mean Profiles by Genre')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'v3_mfcc_profiles.png'), dpi=300)
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    io = DataIO()
    X1, X2, X3, y = io.load_processed_data()
    
    if X1 is not None:
        plot_dir = io.get_path('plots')
        plot_manual_features(X1, y, save_path=plot_dir)
        plot_pca_features(X2, y, save_path=plot_dir)
        if X3 is not None:
            plot_rich_features_profile(X3, y, save_path=plot_dir)