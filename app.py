import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

data_path = 'shopping_trends.csv'
df = pd.read_csv(data_path)

st.title("Analisis Tren Belanja Pelanggan")
st.markdown("### Wawasan dan Visualisasi dari Tren Belanja")

st.sidebar.header("Opsi Filter")
age_range = st.sidebar.slider("Pilih Rentang Usia", int(df['Age'].min()), int(df['Age'].max()), (20, 50))
category_filter = st.sidebar.multiselect("Pilih Kategori", df['Category'].unique(), default=df['Category'].unique())
season_filter = st.sidebar.multiselect("Pilih Musim", df['Season'].unique(), default=df['Season'].unique())

filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
filtered_df = filtered_df[filtered_df['Season'].isin(season_filter)]

st.markdown("### Data yang Difilter")
st.dataframe(filtered_df)

st.markdown("### Distribusi Jumlah Pembelian")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(filtered_df['Purchase Amount (USD)'], bins=20, kde=True, ax=ax)
ax.set_title("Distribusi Jumlah Pembelian (USD)")
st.pyplot(fig)

st.markdown("### Rata-Rata Pembelian per Kategori")
avg_purchase = filtered_df.groupby('Category')['Purchase Amount (USD)'].mean().sort_values(ascending=False)
st.bar_chart(avg_purchase)

st.markdown("### Frekuensi Pembelian per Musim")
season_counts = filtered_df['Season'].value_counts()
st.bar_chart(season_counts)

st.markdown("Dikembangkan menggunakan Streamlit")
