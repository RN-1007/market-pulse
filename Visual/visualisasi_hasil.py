import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Setting gaya visualisasi agar terlihat profesional
plt.style.use('ggplot') 
sns.set(style="whitegrid")

def run_visualization():
    print("1. Membaca Data Hasil Segmentasi...")
    try:
        df = pd.read_csv('hasil_segmentasi_final.csv')
        # Pastikan Customer ID jadi Index jika perlu
        if 'Customer ID' in df.columns:
            df.set_index('Customer ID', inplace=True)
    except FileNotFoundError:
        print("ERROR: File 'hasil_segmentasi_final.csv' tidak ditemukan. Jalankan modeling dulu.")
        return

    print("   Data dimuat. Membuat grafik...")

    # --- GRAFIK 1: PERSEBARAN JUMLAH PELANGGAN (Pie Chart) ---
    print("2. Membuat Grafik Cluster Size...")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    plt.figure(figsize=(7, 7))
    plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index], 
            autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Proporsi Jumlah Pelanggan per Cluster')
    plt.savefig('vis_1_cluster_size.png')
    print("   > Saved: vis_1_cluster_size.png")

    # --- GRAFIK 2: PROFILING (Boxplot R, F, M) ---
    # Ini untuk melihat "Siapa yang paling kaya?" "Siapa yang paling sering datang?"
    print("3. Membuat Grafik Profiling (Boxplot)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(x='Cluster', y='Recency', data=df, ax=axes[0], palette="Set2")
    axes[0].set_title('Recency (Semakin Kecil = Semakin Bagus)')
    
    # Zoom in sedikit jika ada outlier ekstrem agar kotak terbaca
    axes[0].set_ylim(0, df['Recency'].quantile(0.95)) 

    sns.boxplot(x='Cluster', y='Frequency', data=df, ax=axes[1], palette="Set2")
    axes[1].set_title('Frequency (Semakin Besar = Semakin Bagus)')
    axes[1].set_ylim(0, df['Frequency'].quantile(0.95))

    sns.boxplot(x='Cluster', y='Monetary', data=df, ax=axes[2], palette="Set2")
    axes[2].set_title('Monetary (Semakin Besar = Semakin Bagus)')
    axes[2].set_ylim(0, df['Monetary'].quantile(0.95))

    plt.tight_layout()
    plt.savefig('vis_2_profiling_boxplot.png')
    print("   > Saved: vis_2_profiling_boxplot.png")

    # --- GRAFIK 3: SNAKE PLOT (Pola Perilaku) ---
    # Grafik ini butuh data yang di-scale lagi agar bisa digabung dalam satu grafik
    print("4. Membuat Snake Plot...")
    
    # Scaling sementara hanya untuk visualisasi ini
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']]),
                             index=df.index, columns=['Recency', 'Frequency', 'Monetary'])
    df_scaled['Cluster'] = df['Cluster']
    
    # Melt data (mengubah bentuk tabel melebar jadi memanjang ke bawah)
    df_melt = pd.melt(df_scaled.reset_index(), 
                      id_vars=['Customer ID', 'Cluster'],
                      value_vars=['Recency', 'Frequency', 'Monetary'],
                      var_name='Metric', value_name='Value')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='Metric', y='Value', hue='Cluster', 
                 palette='bright', marker='o', linewidth=2.5)
    plt.title('Snake Plot: Pola Perilaku Tiap Cluster')
    plt.xlabel('Metric (RFM)')
    plt.ylabel('Nilai Standar (Z-Score)')
    plt.legend(title='Cluster', loc='upper right')
    plt.savefig('vis_3_snake_plot.png')
    print("   > Saved: vis_3_snake_plot.png")
    
    # --- GRAFIK 4: SCATTER PLOT 3D (BONUS KEREN) ---
    # Visualisasi sebaran Monetary vs Frequency
    print("5. Membuat Scatter Plot (Frequency vs Monetary)...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Frequency', y='Monetary', hue='Cluster', palette='bright', alpha=0.7)
    plt.title('Peta Persebaran: Frequency vs Monetary')
    plt.xlim(0, df['Frequency'].quantile(0.98)) # Zoom in biar tidak kejauhan
    plt.ylim(0, df['Monetary'].quantile(0.98))
    plt.savefig('vis_4_scatter.png')
    print("   > Saved: vis_4_scatter.png")

    print("\nSELESAI! Silakan cek 4 gambar PNG yang muncul.")

if __name__ == "__main__":
    run_visualization()