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
    st.write(dataframe.info())

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

    # Memilih fitur untuk clustering
    st.write("### Pilih Fitur untuk Clustering")
    fitur = st.multiselect("Fitur", options=dataframe.columns, default=['Purchase Amount (USD)', 'Age'])

    if len(fitur) > 1:
        # Memeriksa apakah fitur yang dipilih bertipe numerik
        fitur_numerik = [kolom for kolom in fitur if pd.api.types.is_numeric_dtype(dataframe[kolom])]
        if len(fitur_numerik) < len(fitur):
            st.warning("Beberapa fitur yang dipilih bukan numerik dan akan diabaikan.")

        # Menyiapkan data untuk clustering
        data = dataframe.dropna(subset=fitur_numerik)
        X = data[fitur_numerik]
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
        st.write(pd.DataFrame(kmeans.cluster_centers_, columns=fitur_numerik))

        st.write("### Data dengan Label Cluster")
        st.write(data[['Cluster'] + fitur_numerik].head())

        # Visualisasi cluster jika data 2D
        if len(fitur_numerik) == 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=fitur_numerik[0], y=fitur_numerik[1], hue='Cluster', data=data, palette='tab10', ax=ax)
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)
        else:
            st.write("Visualisasi hanya tersedia untuk 2 fitur.")
    else:
        st.warning("Pilih setidaknya 2 fitur numerik untuk clustering.")
