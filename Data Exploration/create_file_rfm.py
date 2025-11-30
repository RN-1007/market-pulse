import pandas as pd
import datetime as dt

def make_rfm_file():
    print("1. Membaca Data Bersih (online_retail_clean_2.csv)...")
    try:
        # Load data transaksi bersih
        df = pd.read_csv('online_retail_clean_2.csv')
        
        # Pastikan kolom tanggal dikenali sebagai datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Pastikan kolom TotalAmount ada (Quantity * Price)
        # Jaga-jaga jika di file clean belum ada kolom ini
        if 'TotalAmount' not in df.columns:
            df['TotalAmount'] = df['Quantity'] * df['Price']
            
    except FileNotFoundError:
        print("ERROR: File 'online_retail_clean_2.csv' tidak ditemukan.")
        return

    print("2. Melakukan Agregasi ke Format RFM...")
    
    # Menentukan tanggal patokan (1 hari setelah transaksi terakhir di seluruh data)
    target_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    # TRANSFORMASI DATA: Dari Transaksi -> Menjadi Pelanggan
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (target_date - x.max()).days, # Recency: Jarak hari beli terakhir
        'Invoice': 'nunique',                                     # Frequency: Jumlah struk unik
        'TotalAmount': 'sum'                                      # Monetary: Total belanja
    })

    # Ganti nama kolom agar standar
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)

    # Filter kecil: Memastikan Monetary > 0 (kadang ada retur yang lolos)
    rfm = rfm[rfm['Monetary'] > 0]

    print(f"   - Berhasil merangkum menjadi {rfm.shape[0]} pelanggan unik.")
    print("   - Contoh data:")
    print(rfm.head())

    # 3. MENYIMPAN FILE (PENTING!)
    output_file = 'rfm_data.csv'
    rfm.to_csv(output_file)
    print(f"\n3. SUKSES! File '{output_file}' telah tersimpan.")
    print("   Sekarang Anda bisa lanjut ke tahap Feature Engineering.")

if __name__ == "__main__":
    make_rfm_file()