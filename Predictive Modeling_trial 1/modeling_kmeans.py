import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Setting gaya grafik
sns.set(style="whitegrid")

def run_modeling():
    print("1. Membaca Data...")
    try:
        # Load data yang sudah di-scaling (untuk dimakan algoritma)
        df_scaled = pd.read_csv('rfm_siap_model.csv', index_col=0)
        
        # Load data RFM asli (untuk ditempel hasil labelnya nanti)
        # Kita butuh angka aslinya (Rupiah/Dollar) biar bisa dibaca manusia
        df_original = pd.read_csv('rfm_data.csv')
        if 'Customer ID' in df_original.columns:
            df_original.set_index('Customer ID', inplace=True)
            
    except FileNotFoundError:
        print("ERROR: File tidak ditemukan. Pastikan 'rfm_siap_model.csv' dan 'rfm_data.csv' ada.")
        return

    print("   Data siap. Dimensi: ", df_scaled.shape)

    # --- BAGIAN A: ELBOW METHOD (Mencari Jumlah Cluster Optimal) ---
    print("\n2. Menjalankan Elbow Method (Tunggu sebentar)...")
    inertia = []
    k_range = range(1, 11)  # Coba dari 1 sampai 10 cluster
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_) # Inertia = Total Error

    # Plot Grafik Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='b')
    plt.title('Elbow Method: Mencari K Terbaik')
    plt.xlabel('Jumlah Cluster (k)')
    plt.ylabel('Inertia (Error)')
    plt.xticks(k_range)
    plt.grid(True)
    
    # Simpan untuk PPT
    plt.savefig('grafik_elbow.png')
    print("   Grafik Elbow disimpan: 'grafik_elbow.png'")
    
    # --- BAGIAN B: MENJALANKAN K-MEANS FINAL ---
    # Biasanya untuk retail, 3 cluster adalah angka 'magic' (Loyal, Biasa, Hilang)
    # Nanti Anda bisa ganti angka ini sesuai lekukan di grafik elbow
    k_pilihan = 3 
    print(f"\n3. Menjalankan K-Means dengan {k_pilihan} Cluster...")
    
    model = KMeans(n_clusters=k_pilihan, random_state=42, n_init=10)
    model.fit(df_scaled)
    
    # Mendapatkan Label (0, 1, 2)
    labels = model.labels_
    
    # Tempelkan label ke data ASLI (Bukan data scaled)
    df_original['Cluster'] = labels
    
    # --- BAGIAN C: LIHAT HASILNYA ---
    print("\n4. Hasil Clustering (Rata-rata per Kelompok):")
    summary = df_original.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count' # Hitung jumlah orang
    }).rename(columns={'Cluster': 'Jumlah Anggota'})
    
    print(summary)
    # Tips: Copy tabel yang muncul di terminal ini ke PPT bagian 'Result'

    # Simpan hasil final
    df_original.to_csv('hasil_final_segmentasi.csv')
    print("\nSUKSES! File hasil akhir disimpan: 'hasil_final_segmentasi.csv'")

if __name__ == "__main__":
    run_modeling()