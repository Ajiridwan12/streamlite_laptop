import pickle
import streamlit as st
import setuptools

model = pickle.load(open('estimasi_harga_laptop1.sav', 'rb'))

st.title('Estimasi Harga Laptop')

Ram = st.number_input('Masukan ram')
Weight= st.number_input('Masukan berat laptop')
Ips = st.number_input('Masukan Ips')
Ppi = st.number_input('masukan Ppi')
HDD = st.number_input('Masukan HDD')
SSD = st.number_input('Masukan SSD')
Cpu_brand = st.number_input('Masukkan brand Cpu (AMD = 1, Intel I3 = 3, Intel I5 = 5, Intel I7 = 7, Other = 0)')
Gpu_brand = st.number_input('Masukan Brand GPU (Intel = 0, AMD = 1, Nvidia = 2)')
Os = st.number_input ('Masukkan OS Laptop (Mac = 0, Windows = 1, Others = 2)')

predict = ''

if st.button('Estimasi Harga'):
        predict = model.predict([[Ram,Weight,Ips,Ppi,Cpu_brand,HDD,SSD,Gpu_brand,Os]])

        st.success(f'Estimasi Harga : {predict[0]:.2f}')