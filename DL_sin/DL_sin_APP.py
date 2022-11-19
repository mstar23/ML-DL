import streamlit as st
import joblib
import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt

st.write('## sin함수 예측하기')
st.write(
    '''
    colab 노트북 주소 : https://colab.research.google.com/drive/1NNg3VC674PrIFBVcYJcVepeWGZc6JkEe?usp=sharing
    '''
)

st.write('torch 연결 완료?')
st.write('갈 길이 멀다;;')