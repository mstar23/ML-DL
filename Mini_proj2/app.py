import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt

st.write('##  랜덤포레스트를 이용한 비행기 가격 예측')
st.write('---')

# 데이터 불러오기
data_path = f"{os.path.dirname(os.path.abspath(__file__))}/data.csv"
data = pd.read_csv(data_path)
# df = pd.DataFrame(data)
st.write(data)