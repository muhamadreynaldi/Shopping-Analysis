import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Memuat dataset
path_data = 'shopping_trends.csv'  # Ganti dengan path dataset Anda
dataframe = pd.read_csv(path_data)

# Aplikasi Streamlit
st.title("Analisis Tren Belanja & K-Means Clustering")

# Menu Sidebar
menu = st.sidebar.selectbox("Menu", ["Gambaran Umum", "Visualisasi", "K-Means Clustering"])

if menu == "Gambaran Umum":
    st.subheader("Gambaran Umum Dataset")
    st.write("### 5 Baris Pertama Dataset")
    st.write(dataframe.head())

    st.write("### Informasi Dataset")
    df_info = pd.DataFrame({
        "Column": dataframe.columns,
        "Non-Null Count": dataframe.notnull().sum(),
        "Dtype": dataframe.dtypes
    }).reset_index(drop=True)
    st.write(df_info)

    st.write("### Nilai yang Hilang")
    st.write(dataframe.isnull().sum())

if menu == "Visualisasi":
    st.subheader("Visualisasi")

    # Distribusi item yang dibeli
    if st.checkbox("Tampilkan Distribusi Item yang Dibeli"):
        st.write("### Distribusi Item yang Dibeli")
        fig, ax = plt.subplots(figsize=(10, 6))
        dataframe['Item Purchased'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        ax.set_title("Distribusi Item yang Dibeli")
        st.pyplot(fig)

    # Jumlah pembelian per item
    if st.checkbox("Tampilkan Total Jumlah Pembelian per Item"):
        st.write("### Total Jumlah Pembelian per Item")
        item = dataframe.groupby('Item Purchased')['Purchase Amount (USD)'].sum().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=item, x='Item Purchased', y='Purchase Amount (USD)', ax=ax, palette='viridis')
        ax.set_title("Total Jumlah Pembelian per Item")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

    # Boxplot usia
    if st.checkbox("Tampilkan Distribusi Usia"):
        df_usia = dataframe.dropna(subset=['Age'])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df_usia['Age'], color='lightcoral', ax=ax)
        ax.set_title("Boxplot Usia")
        st.pyplot(fig)

if menu == "K-Means Clustering":
    st.subheader("K-Means Clustering")

    # Memilih rentang usia
    st.write("### Pilih Rentang Usia")
    min_age = int(dataframe['Age'].min())
    max_age = int(dataframe['Age'].max())
    age_range = st.slider("Rentang Usia", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Filter data berdasarkan rentang usia
    data_filtered = dataframe[(dataframe['Age'] >= age_range[0]) & (dataframe['Age'] <= age_range[1])]

    if data_filtered.empty:
        st.warning("Tidak ada data dalam rentang usia yang dipilih.")
    else:
        # Memilih fitur untuk clustering (hanya numerik)
        st.write("### Pilih Fitur untuk Clustering")
        fitur_numerik = [kolom for kolom in data_filtered.columns if pd.api.types.is_numeric_dtype(data_filtered[kolom])]
        fitur = st.multiselect("Fitur", options=fitur_numerik, default=['Purchase Amount (USD)', 'Age'])

        if len(fitur) > 1:
            # Menyiapkan data untuk clustering
            data = data_filtered.dropna(subset=fitur)
            X = data[fitur]
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
            except ValueError as e:
                st.error(f"Kesalahan saat scaling: {e}")
                st.stop()

            # Memilih jumlah cluster
            st.write("### Pilih Jumlah Cluster")
            n_cluster = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)

            # Menerapkan K-Means
            kmeans = KMeans(n_clusters=n_cluster, random_state=42)
            data['Cluster'] = kmeans.fit_predict(X_scaled)

            # Menampilkan hasil clustering
            st.write("### Pusat Cluster")
            st.write(pd.DataFrame(kmeans.cluster_centers_, columns=fitur))

            st.write("### Data dengan Label Cluster")
            st.write(data[['Cluster'] + fitur].head())

            # Visualisasi cluster untuk kombinasi dua fitur
            st.write("### Visualisasi untuk Kombinasi Fitur")
            if len(fitur) >= 2:
                feature_pairs = [(fitur[i], fitur[j]) for i in range(len(fitur)) for j in range(i + 1, len(fitur))]
                for pair in feature_pairs:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=pair[0], y=pair[1], hue='Cluster', data=data, palette='tab10', ax=ax)
                    ax.set_title(f"K-Means Clustering: {pair[0]} dan {pair[1]}")
                    st.pyplot(fig)

            # Kesimpulan otomatis
            st.write("### Kesimpulan")
            def generate_summary(cluster_data, fitur):
                summaries = []
                for cluster_id, subset in cluster_data.groupby('Cluster'):
                    summary = f"Cluster {cluster_id}:\n"
                    interpretations = []
                    for col in fitur:
                        mean_value = round(subset[col].mean(), 2)
                        min_value = round(subset[col].min(), 2)
                        max_value = round(subset[col].max(), 2)
                        summary += (f"  - {col}:\n"
                                    f"    - Rata-rata: {mean_value}\n"
                                    f"    - Rentang: ({min_value} - {max_value})\n")
                        # Interpretasi tambahan untuk fitur tertentu
                        if col == 'Purchase Amount (USD)':
                            if mean_value > 50.1:
                                interpretations.append("Pelanggan dengan pengeluaran tinggi.")
                            elif mean_value < 50:
                                interpretations.append("Pelanggan dengan pengeluaran rendah.")
                        elif col == 'Age':
                            if mean_value < 44:
                                interpretations.append("Cluster ini didominasi oleh pelanggan muda.")
                            elif mean_value > 44:
                                interpretations.append("Cluster ini mencerminkan pelanggan dewasa.")
                        elif col == 'Review Rating':
                            if mean_value >= 3.71:
                                interpretations.append("Cluster ini mencerminkan kepuasan pelanggan yang tinggi.")
                            elif mean_value <= 3.7:
                                interpretations.append("Cluster ini mencerminkan kepuasan pelanggan yang rendah.")
                        elif col == 'Previous Purchases':
                            if mean_value >= 26:
                                interpretations.append("Pelanggan dengan riwayat pembelian tinggi.")
                            elif mean_value < 25:
                                interpretations.append("Pelanggan dengan riwayat pembelian rendah.")
                    if interpretations:
                        summary += "  - Interpretasi:\n    - " + "\n    - ".join(interpretations) + "\n"
                    summaries.append(summary)
                return summaries

            summaries = generate_summary(data, fitur)
            for summary in summaries:
                st.text(summary)
        else:
            st.warning("Pilih setidaknya 2 fitur numerik untuk clustering.")
