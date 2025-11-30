import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Setting gaya visualisasi
sns.set(style="whitegrid")

def run_smart_modeling():
    print("==============================================")
    print("   MODUL K-MEANS: TRIAL & ERROR AUTOMATION    ")
    print("==============================================")
    
    # --- 1. MEMBACA DATA ---
    print("\n[PHASE 1] Membaca Data...")
    try:
        # Load data scaled (untuk algoritma)
        df_scaled = pd.read_csv('rfm_siap_model.csv', index_col=0)
        # Load data asli (untuk label hasil akhir)
        df_original = pd.read_csv('rfm_data.csv')
        if 'Customer ID' in df_original.columns:
            df_original.set_index('Customer ID', inplace=True)
            
        print(f"   > Data berhasil dimuat. Total Pelanggan: {df_scaled.shape[0]}")
    except FileNotFoundError:
        print("   > ERROR: File tidak ditemukan. Pastikan 'rfm_siap_model.csv' & 'rfm_data.csv' ada.")
        return

    # --- 2. TRIAL & ERROR (Mencari K Terbaik) ---
    print("\n[PHASE 2] Memulai Proses Trial & Error (Iterasi Cluster)...")
    print("   > Algoritma akan mencoba membagi pelanggan jadi 2 s/d 6 kelompok.")
    print("   > Parameter: Inertia (Error terendah) & Silhouette (Kepadatan terbaik).")
    print("-" * 60)
    
    inertia_list = []
    silhouette_list = []
    k_range = range(2, 7)  # Kita coba k=2, 3, 4, 5, 6
    
    best_score = -1
    best_k = 0

    for k in k_range:
        # Jalankan K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        
        # Hitung Metrik
        inertia = kmeans.inertia_
        score = silhouette_score(df_scaled, kmeans.labels_)
        
        # Simpan hasil
        inertia_list.append(inertia)
        silhouette_list.append(score)
        
        # Cek apakah ini skor terbaik sejauh ini?
        if score > best_score:
            best_score = score
            best_k = k
            
        print(f"   [TRIAL] Testing k={k} cluster... | Inertia: {inertia:.0f} | Silhouette Score: {score:.4f}")

    print("-" * 60)
    print(f"   > HASIL TRIAL: Jumlah cluster terbaik adalah k={best_k} (Score: {best_score:.4f})")
    print("   > Sistem otomatis memilih k tersebut untuk model final.")

    # --- 3. VISUALISASI PROSES TRIAL (GRAFIK) ---
    print("\n[PHASE 3] Menyimpan Grafik Evaluasi...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Grafik Elbow (Inertia) - Garis Biru
    color = 'tab:blue'
    ax1.set_xlabel('Jumlah Cluster (k)')
    ax1.set_ylabel('Inertia (Semakin Kecil Semakin Bagus)', color=color)
    ax1.plot(k_range, inertia_list, marker='o', color=color, label='Inertia (Elbow)')
    ax1.tick_params(axis='y', labelcolor=color)

    # Grafik Silhouette (Score) - Garis Merah
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score (Semakin Besar Semakin Bagus)', color=color)
    ax2.plot(k_range, silhouette_list, marker='x', linestyle='--', color=color, label='Silhouette Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Evaluasi Trial & Error (Pemenang: k={best_k})')
    fig.tight_layout()
    plt.savefig('grafik_evaluasi_trial.png')
    print("   > Grafik disimpan: 'grafik_evaluasi_trial.png'")

    # --- 4. FINAL MODELING ---
    print(f"\n[PHASE 4] Membuat Model Final dengan {best_k} Cluster...")
    
    # Jalankan ulang K-Means dengan k pemenang
    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_model.fit(df_scaled)
    
    # Ambil labelnya
    df_original['Cluster'] = final_model.labels_
    
    # --- 5. REPORTING HASIL ---
    print("\n[PHASE 5] Ringkasan Profil Pelanggan:")
    
    # Hitung rata-rata per cluster
    summary = df_original.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Jumlah Anggota'})
    
    print(summary)
    
    # Simpan File Akhir
    output_csv = 'hasil_segmentasi_final.csv'
    df_original.to_csv(output_csv)
    print(f"\n   > SUKSES! Data hasil segmentasi disimpan ke '{output_csv}'")
    print("==============================================")

if __name__ == "__main__":
    run_smart_modeling()