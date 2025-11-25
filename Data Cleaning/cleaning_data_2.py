import pandas as pd
import os

def clean_retail_data():
    # Sesuaikan nama file input kamu
    input_file = 'online_retail_clean.csv' 
    output_file = 'online_retail_clean_2.csv'

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
    
    rows_initial = df.shape[0]
    print(f"   Data Awal: {rows_initial} baris, {df.shape[1]} kolom")

    # --- PROSES CLEANING ---
    print("\n2. Memulai proses pembersihan...")
    
    # 1. Hapus baris tanpa Customer ID
    df_clean = df.dropna(subset=['Customer ID']).copy()
    
    # 2. Hapus Duplikat
    df_clean = df_clean.drop_duplicates()
    
    # 3. Filter: Hanya Ambil Quantity > 0 dan Price > 0
    # (Membuang pembatalan 'C' dan error input/minus)
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]

    # --- [BARU] FILTER KODE SAMPAH OPERASIONAL ---
    # Membuang baris yang bukan produk (Ongkir, Manual, Diskon, dll)
    sampah_operasional = ['POST', 'M', 'BANK CHARGES', 'PADS', 'D', 'DOT', 'CRUK']
    df_clean = df_clean[~df_clean['StockCode'].isin(sampah_operasional)]
    print(f"   - Membersihkan kode non-produk: {', '.join(sampah_operasional)}")
    # ---------------------------------------------
    
    # 4. Ubah Tipe Data
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
    
    # 5. Feature Engineering: Tambah kolom Total Belanja
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['Price']

    # Hitung statistik pembersihan
    rows_final = df_clean.shape[0]
    rows_deleted = rows_initial - rows_final

    print(f"   Data Bersih: {rows_final} baris")
    print(f"   Dibuang: {rows_deleted} baris (Data kosong, duplikat, retur, & kode sampah).")

    # --- SIMPAN HASIL ---
    print(f"\n3. Menyimpan data bersih ke '{output_file}'...")
    df_clean.to_csv(output_file, index=False)
    print("SELESAI! File siap digunakan untuk analisis Market Pulse.")

    # Tampilkan sedikit preview
    print("\nPreview Data Bersih:")
    print(df_clean[['Invoice', 'StockCode', 'TotalAmount', 'InvoiceDate']].head())

if __name__ == "__main__":
    clean_retail_data()