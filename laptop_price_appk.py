import pandas as pd   # data preprocessing
import numpy as np    # mathematical computation
from sklearn import *
import pickle
import streamlit as st


# Load the model and dataset
model = pickle.load(open('dt_model.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

st.title('Laptop Price Prediction')
st.header('Fill the details to predict laptop Price')


# Company         object
# TypeName        object
# Ram              int64
# Weight         float64
# Price          float64
# Touchscreen      int64
# Ips              int64
# Cpu brand       object
# HDD              int64
# SSD              int64
# Gpu brand       object
# os              object

company = st.selectbox('Company',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('Ram in (GB)',[8, 16, 4,2, 12,6, 32,24,64])
weight = st.number_input('Weight(in kg)')
touchscreen = st.selectbox('Touchscreen',['No','Yes'])  # actual value is [0,1]
ips = st.selectbox('IPS',['No','Yes'])              # actual value is [0,1]
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
hdd = st.selectbox('HDD(GB)', [0,  500, 1000, 2000,   32,  128])
ssd = st.selectbox('SSD(GB)',[128, 0,256,512,32,64,1000,1024,16,768,180,240,8])
gpu = st.selectbox('GPU',df['Gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())


if st.button('Predict Laptop Price'):
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    if ips=='Yes':
        ips=1
    else:
        ips=0
    test = np.array([company,type,ram,weight,touchscreen,ips,cpu,hdd,ssd,gpu,os])
    test = test.reshape([1,11])

    st.success(model.predict(test)[0])









