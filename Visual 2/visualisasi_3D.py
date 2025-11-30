import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def show_3d_cluster():
    print("Membaca data hasil segmentasi...")
    try:
        df = pd.read_csv('hasil_segmentasi_final.csv')
    except FileNotFoundError:
        print("File hasil segmentasi tidak ditemukan.")
        return

    print("Membuat Plot 3D...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot setiap titik (gunakan sample jika data terlalu besar)
    import numpy as np
    
    if len(df) > 1000:
        df_sample = df.sample(1000, random_state=42)
    else:
        df_sample = df

    # Plotting
    scatter = ax.scatter(np.log1p(df_sample['Recency']), 
                         np.log1p(df_sample['Frequency']), 
                         np.log1p(df_sample['Monetary']), 
                         c=df_sample['Cluster'], 
                         cmap='viridis', 
                         s=40, alpha=0.6)

    # Label Sumbu
    ax.set_xlabel('Recency (Log)')
    ax.set_ylabel('Frequency (Log)')
    ax.set_zlabel('Monetary (Log)')
    ax.set_title('Visualisasi Cluster 3D (Recency, Frequency, Monetary)')

    # Tambahkan Legenda
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    plt.savefig('visualisasi_3d.png')
    print("Grafik 3D disimpan sebagai 'visualisasi_3d.png'")
    plt.show()

if __name__ == "__main__":
    show_3d_cluster()