import streamlit as st
# import joblib
import pandas as pd # 판다스 불러오기

from PIL import Image # 파이썬 기본라이브러리는 바로 사용 가능!
import os
def get_image(image_name):
    image_path = f"{os.path.dirname(os.path.abspath(__file__))}/{image_name}"
    image = Image.open(image_path) # 경로와 확장자 주의!
    st.image(image)

#### 프로젝트 네임 #####
st.write('# LGBM 실습 페이지')
st.write('집에서 수정중')
# #### 데이터 불러오기 ####
# st.write('#### 전처리한 데이터')
# model_train_joblib = joblib.load('lgbm_df.pkl')
# st.write(lgbm_df.head())

### 불러온 데이터로 훈련-테스트셋 분리

## joblib 설치 했는데 실행하는법 + 대용량 pkl 파일 guthub 업로드 방법???

