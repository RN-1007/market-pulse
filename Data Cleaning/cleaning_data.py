import pandas as pd
import os

def clean_retail_data():
    input_file = 'online_retail_II.csv'
    output_file = 'online_retail_clean.csv'

    # Cek apakah file ada
    if not os.path.exists(input_file):
        print(f"ERROR: File '{input_file}' tidak ditemukan di folder ini.")
        return

    print("1. Sedang membaca data... (Mungkin butuh waktu beberapa detik)")
    try:
        df = pd.read_csv(input_file)
    except UnicodeDecodeError:
        print("   - Mendeteksi encoding berbeda, mencoba ISO-8859-1...")
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
    
    print(f"   Data Awal: {df.shape[0]} baris, {df.shape[1]} kolom")

    # --- PROSES CLEANING ---
    print("\n2. Memulai proses pembersihan...")
    
    # Hapus baris tanpa Customer ID
    df_clean = df.dropna(subset=['Customer ID']).copy()
    
    # Hapus Duplikat
    df_clean = df_clean.drop_duplicates()
    
    # Filter: Hanya Ambil Quantity > 0 dan Price > 0
    # (Membuang pembatalan 'C' dan error input)
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
    
    # Ubah Tipe Data
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
    
    # Feature Engineering: Tambah kolom Total Belanja
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']

    print(f"   Data Bersih: {df_clean.shape[0]} baris")
    print(f"   Dibuang: {df.shape[0] - df_clean.shape[0]} baris sampah/tidak valid.")

    # --- SIMPAN HASIL ---
    print(f"\n3. Menyimpan data bersih ke '{output_file}'...")
    df_clean.to_csv(output_file, index=False)
    print("SELESAI! File siap digunakan untuk analisis selanjutnya.")

    # Tampilkan sedikit preview
    print("\nPreview Data:")
    print(df_clean[['Invoice', 'Customer ID', 'TotalAmount', 'InvoiceDate']].head())

if __name__ == "__main__":
    clean_retail_data()