import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Setting tampilan grafik
sns.set(style="whitegrid")

def run_feature_engineering():
    print("1. Membaca file 'rfm_data.csv'...")
    try:
        rfm = pd.read_csv('rfm_data.csv')
        # Jadikan Customer ID sebagai Index (agar tidak ikut dihitung rumusnya)
        if 'Customer ID' in rfm.columns:
            rfm.set_index('Customer ID', inplace=True)
    except FileNotFoundError:
        print("ERROR: File 'rfm_data.csv' belum ada. Jalankan kode langkah sebelumnya dulu.")
        return

    print("   Data dimuat. Melakukan transformasi...")

    # --- LANGKAH A: Log Transformation ---
    # Mengatasi masalah skewness (data miring)
    # Kita pakai log1p (logaritma x + 1) untuk antisipasi jika ada angka 0
    rfm_log = np.log1p(rfm)

    # --- LANGKAH B: Scaling (StandardScaler) ---
    # Menyamakan skala data (Mean=0, Std=1) agar K-Means adil
    scaler = StandardScaler()
    rfm_scaled_matrix = scaler.fit_transform(rfm_log)
    
    # Kembalikan ke bentuk Tabel (DataFrame) dengan nama kolom yang benar
    rfm_scaled = pd.DataFrame(rfm_scaled_matrix, index=rfm.index, columns=rfm.columns)

    print("   Data berhasil di-transformasi.")

    # --- LANGKAH C: Visualisasi Before vs After (PENTING BUAT PPT) ---
    print("2. Membuat grafik perbandingan...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Baris 1: SEBELUM (Data Asli)
    sns.histplot(rfm['Recency'], ax=axes[0, 0], kde=True, color='red').set_title('SEBELUM: Recency (Miring)')
    sns.histplot(rfm['Frequency'], ax=axes[0, 1], kde=True, color='red').set_title('SEBELUM: Frequency (Miring)')
    sns.histplot(rfm['Monetary'], ax=axes[0, 2], kde=True, color='red').set_title('SEBELUM: Monetary (Miring)')

    # Baris 2: SESUDAH (Data Log + Scaled)
    sns.histplot(rfm_scaled['Recency'], ax=axes[1, 0], kde=True, color='blue').set_title('SESUDAH: Recency (Normal)')
    sns.histplot(rfm_scaled['Frequency'], ax=axes[1, 1], kde=True, color='blue').set_title('SESUDAH: Frequency (Normal)')
    sns.histplot(rfm_scaled['Monetary'], ax=axes[1, 2], kde=True, color='blue').set_title('SESUDAH: Monetary (Normal)')

    plt.tight_layout()
    plt.savefig('grafik_perbandingan_feature.png')
    print("   Grafik disimpan: 'grafik_perbandingan_feature.png'")

    # --- LANGKAH D: Simpan Data Siap Model ---
    output_file = 'rfm_siap_model.csv'
    rfm_scaled.to_csv(output_file)
    print(f"\n3. SELESAI! Data siap modeling disimpan ke '{output_file}'")
    
    print("\nContoh 5 baris data yang sudah di-scale:")
    print(rfm_scaled.head())

if __name__ == "__main__":
    run_feature_engineering()