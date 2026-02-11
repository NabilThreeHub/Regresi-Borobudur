import streamlit as st
import pandas as pd
import joblib

model= joblib.load("model_regresi_borobudur.joblib")

st.sidebar.title("Informasi")
st.sidebar.success("Dibuat oleh Nabil Albara")

st.set_page_config(
	page_title="Regresi Pengunjung Borodudur"
)

st.title("Memprediksi Pengunjung di Borobudur")
st.markdown("Machine Learning Regresi yang memprediksi pengunjung harian di Borobudur dengan menggunakan data tipe hari, musim, suhu rata rata, event budaya, harga tiket")

hari_type = st.selectbox("Hari", ["weekday", "weekend"])
musim = st.selectbox("Cuaca", ["kemarau", "hujan"])
suhu_rata_rata = st.slider("Suhu", 20.10, 35.00, 26.80) 
ada_event_budaya = st.selectbox("Event Budaya", ["ya", "tidak"])
harga_tiket_ribu = st.slider("Harga Tiket", 50.20, 100.00, 74.75) 

if st.button("prediksi"):
	data_baru=pd.DataFrame([[hari_type, musim, suhu_rata_rata, ada_event_budaya, harga_tiket_ribu]],
        	columns=["hari_type", "musim", "suhu_rata_rata", "ada_event_budaya", "harga_tiket_ribu"])
	prediksi=model.predict(data_baru)[0]
	st.success(f"Jumlah Pengujung = {prediksi:.0f}")
	st.balloons()

st.divider()
st.caption("Dibuat oleh Nabil Albara")