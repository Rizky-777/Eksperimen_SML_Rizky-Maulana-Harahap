import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def run_preprocessing():
    print("⏳ Memulai otomatisasi Data Preprocessing...")

    # 1. Memuat Dataset Mentah
    df = pd.read_csv('../data.csv')

    # 2. Memisahkan Fitur (X) dan Target (y)
    X = df.drop('Creditworthiness', axis=1)
    y = df['Creditworthiness']

    # 3. Encoding Kategorikal (Ubah teks ke angka)
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # 4. Scaling Numerik (Menyamakan skala angka)
    numeric_cols = X.columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menggabungkan kembali jadi DataFrame
    df_final = pd.DataFrame(X_scaled, columns=numeric_cols)
    df_final['Creditworthiness'] = y.values # Masukkan kembali targetnya

    # 5. Menyimpan hasil preprocessing ke folder baru
    output_dir = 'dataset_preprocessing'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'data_preprocessed.csv')
    
    df_final.to_csv(output_path, index=False)
    print(f"✅ Preprocessing Berhasil! Data bersih disimpan di: {output_path}")

if __name__ == "__main__":
    run_preprocessing()