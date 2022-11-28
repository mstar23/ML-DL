import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
import os
# import plotly.plotly as py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.write('##  랜덤포레스트를 이용한 비행기 가격 예측')
st.write('---')

# 데이터 불러오기
st.subheader('**이용할 data 상위 5개**')
data_path = f"{os.path.dirname(os.path.abspath(__file__))}/data.csv"
data = pd.read_csv(data_path)
df = pd.DataFrame(data)
df.drop('Unnamed: 0',axis=1,inplace=True)
st.write(df.head())

# 데이터 전처리
X = df.drop('Price', axis=1)
y = df.Price
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100)

# score 와 mse 비교
model_pkl_path = f"{os.path.dirname(os.path.abspath(__file__))}/randomforest.pkl"
model = joblib.load(model_pkl_path)
st.write(model.score(X_train, y_train),model.score(X_test, y_test))
st.subheader('RMSE 비교')
train_pred = model.predict(X_train) 
test_pred = model.predict(X_test) 
train_relation_square = model.score(X_train, y_train)
test_relation_square = model.score(X_test, y_test)
st.write(f' train 결정계수 : {train_relation_square}, test 결정계수 : {test_relation_square}')

# 시각화 해보기

fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers',name='Train'))
fig.add_trace(go.Scatter(x=y_test,y=test_pred,mode='markers',
              name='Test')) # mode='lines+markers'

fig.update_layout(title='<b>actual과 predict 비교')
fig.show()