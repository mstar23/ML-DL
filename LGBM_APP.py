import streamlit as st

from PIL import Image # 파이썬 기본라이브러리는 바로 사용 가능!
import os
def get_image(image_name):
    image_path = f"{os.path.dirname(os.path.abspath(__file__))}/{image_name}"
    image = Image.open(image_path) # 경로와 확장자 주의!
    st.image(image)

#### 프로젝트 네임 #####
st.write('# LGBM 실습 페이지')
import pandas as pd # 판다스 불러오기
data_url = 'https://media.githubusercontent.com/media/musthave-ML10/data_source/main/fraud.csv'
df = pd.read_csv(data_url) # URL로 CSV 불러오기
st.write('### 사용한 데이터')
st.write(df.head())

