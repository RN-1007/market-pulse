import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_pca_visualization():
    print("1. Membaca Data Hasil Segmentasi...")
    try:
        df = pd.read_csv('hasil_segmentasi_final.csv')
    except FileNotFoundError:
        print("ERROR: File 'hasil_segmentasi_final.csv' tidak ditemukan.")
        return

    print("   Data dimuat. Melakukan Reduksi Dimensi (PCA)...")

    # --- LANGKAH A: Preprocessing (Wajib sama dengan saat Modeling) ---
    # PCA sangat sensitif terhadap skala, jadi kita harus Log + Scale lagi
    features = ['Recency', 'Frequency', 'Monetary']
    
    # 1. Log Transform
    df_log = np.log1p(df[features])
    
    # 2. Scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_log)

    # --- LANGKAH B: Menjalankan PCA ---
    # Mereduksi dari 3 Fitur -> 2 Komponen Utama (PC1 & PC2)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    # Masukkan hasil koordinat baru ke DataFrame
    df['PC1'] = pca_result[:, 0]
    df['PC2'] = pca_result[:, 1]

    # --- LANGKAH C: Visualisasi Scatter Plot 2D ---
    print("2. Membuat Grafik PCA 2D...")
    plt.figure(figsize=(10, 8))
    
    # Plotting titik-titik
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, 
                    palette='bright', s=60, alpha=0.7, edgecolor='k')
    
    # Mempercantik grafik
    plt.title('Visualisasi Cluster 2D menggunakan PCA\n(Reduksi dari Recency, Frequency, Monetary)', fontsize=14)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Cluster', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('vis_pca_2d.png')
    print("   > GRAFIK DISIMPAN: 'vis_pca_2d.png'")
    
    # Penjelasan Variance
    total_var = np.sum(pca.explained_variance_ratio_) * 100
    print(f"\nINFO: Grafik 2D ini merepresentasikan {total_var:.2f}% informasi dari data asli.")
    print("      (Semakin mendekati 100%, semakin akurat gambarnya).")

if __name__ == "__main__":
    run_pca_visualization()