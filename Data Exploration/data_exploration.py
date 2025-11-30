import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Mengatur gaya visualisasi agar terlihat akademis dan rapi
sns.set(style="whitegrid")

def run_exploration():
    # ==========================================
    # 1. PERSIAPAN DATA (Membuat Data RFM)
    # ==========================================
    print("--- 1. MEMBACA & MENYIAPKAN DATA ---")
    try:
        df = pd.read_csv('online_retail_clean_2.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    except FileNotFoundError:
        print("Error: File 'online_retail_clean_2.csv' tidak ditemukan.")
        return

    # Membuat RFM (Recency, Frequency, Monetary)
    # Patokan tanggal analisis = 1 hari setelah transaksi terakhir
    analysis_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days, # Recency
        'Invoice': 'nunique',                                     # Frequency
        'TotalAmount': 'sum'                                      # Monetary
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)
    
    print(f"Data RFM Siap. Jumlah Pelanggan: {rfm.shape[0]}")

    # ==========================================
    # 2. STATISTIK DESKRIPTIF (Untuk Tabel di PPT)
    # ==========================================
    print("\n--- 2. STATISTIK DESKRIPTIF ---")
    stats = rfm.describe()
    print(stats)
    # Tips: Copy hasil tabel di terminal ini ke slide PPT Anda!
    # Ini menunjukkan rata-rata belanja (mean) dan belanja maksimal (max).

    # ==========================================
    # 3. VISUALISASI DISTRIBUSI (Untuk Bukti Data Miring)
    # ==========================================
    print("\n--- 3. MENYIMPAN GRAFIK DISTRIBUSI ---")
    plt.figure(figsize=(15, 5))

    # Grafik Recency
    plt.subplot(1, 3, 1)
    sns.histplot(rfm['Recency'], kde=True, color='skyblue')
    plt.title('Distribusi Recency (Hari)')
    plt.xlabel('Hari sejak pembelian terakhir')

    # Grafik Frequency
    plt.subplot(1, 3, 2)
    sns.histplot(rfm['Frequency'], kde=True, color='orange')
    plt.title('Distribusi Frequency (Kali)')
    plt.xlabel('Jumlah Transaksi')
    plt.xlim(0, 50) # Zoom in agar grafik terbaca (karena banyak yang cuma beli 1x)

    # Grafik Monetary
    plt.subplot(1, 3, 3)
    sns.histplot(rfm['Monetary'], kde=True, color='green')
    plt.title('Distribusi Monetary (Total Belanja)')
    plt.xlabel('Total Belanja (Rupiah/Dollar)')
    plt.xlim(0, 10000) # Zoom in

    plt.tight_layout()
    plt.savefig('grafik_1_distribusi.png')
    print("Berhasil: Grafik 'grafik_1_distribusi.png' disimpan.")

    # ==========================================
    # 4. VISUALISASI OUTLIERS (Untuk Deteksi "Sultan")
    # ==========================================
    print("\n--- 4. MENYIMPAN GRAFIK OUTLIERS (BOXPLOT) ---")
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(y=rfm['Recency'], color='skyblue')
    plt.title('Outliers Recency')

    plt.subplot(1, 3, 2)
    sns.boxplot(y=rfm['Frequency'], color='orange')
    plt.title('Outliers Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(y=rfm['Monetary'], color='green')
    plt.title('Outliers Monetary')

    plt.tight_layout()
    plt.savefig('grafik_2_outliers.png')
    print("Berhasil: Grafik 'grafik_2_outliers.png' disimpan.")

    # ==========================================
    # 5. CEK KORELASI (Opsional, tapi bagus untuk PPT)
    # ==========================================
    print("\n--- 5. MENYIMPAN GRAFIK KORELASI ---")
    plt.figure(figsize=(6, 5))
    sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelasi Antar Variabel RFM')
    plt.savefig('grafik_3_korelasi.png')
    print("Berhasil: Grafik 'grafik_3_korelasi.png' disimpan.")
    
    print("\nSELESAI! Silakan cek 3 gambar PNG yang muncul di folder Anda.")

if __name__ == "__main__":
    run_exploration()