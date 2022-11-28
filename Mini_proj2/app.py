# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import math
# import torch
# import matplotlib.pyplot as plt
# import os
# # import plotly.plotly as py
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import plotly.express as px
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

# st.write('##  랜덤포레스트를 이용한 비행기 가격 예측')
# st.write('---')

# # 데이터 불러오기
# st.subheader('**이용할 data 상위 5개**')
# data_path = f"{os.path.dirname(os.path.abspath(__file__))}/data.csv"
# data = pd.read_csv(data_path)
# df = pd.DataFrame(data)
# df.drop('Unnamed: 0',axis=1,inplace=True)
# st.write(df.head())

# # 데이터 전처리
# X = df.drop('Price', axis=1)
# y = df.Price
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=100)

# # score 와 mse 비교
# model_pkl_path = f"{os.path.dirname(os.path.abspath(__file__))}/randomforest.pkl"
# model = joblib.load(model_pkl_path)
# st.write(model.score(X_train, y_train),model.score(X_test, y_test))
# st.subheader('RMSE 비교')
# train_pred = model.predict(X_train) 
# test_pred = model.predict(X_test) 
# train_relation_square = model.score(X_train, y_train)
# test_relation_square = model.score(X_test, y_test)
# st.write(f' train 결정계수 : {train_relation_square}, test 결정계수 : {test_relation_square}')

# # 시각화 해보기

# fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
# fig.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers',symbol='pentagon',name='Train'))
# fig.add_trace(go.Scatter(x=y_test,y=test_pred,mode='markers',symbol='circle',
#               name='Test')) # mode='lines+markers'

# fig.update_layout(title='<b>actual과 predict 비교')
# st.plotly_chart(fig)

import streamlit as st
# from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt 
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import time
from PIL import Image     # 이미지 처리 라이브러리


########### function ###########

## 데이터 전처리
def preprocessing(df):
    ### 1. Route Drop 처리
    df.drop('Route', axis=1, inplace=True)

    ### 2. Duration 컬럼을 '시간'과 '분' 단위로 분할 후 Duration 컬럼 drop
    df['Dep_Time'] = pd.to_datetime(df['Dep_Time'], format= '%H:%M').dt.time
    df['Duration_hour'] = df.Duration.str.extract('(\d+)h')
    df['Duration_min'] = df.Duration.str.extract('(\d+)m').fillna(0)
    df.drop('Duration', axis=1, inplace=True)
    df.drop(index=6474,inplace=True)

    df.Duration_hour = df.Duration_hour.astype('int64')
    df.Duration_min = df.Duration_min.astype('int64')
    df.Duration_hour = df.Duration_hour*60
    df['Duration_total'] = df.Duration_hour+df.Duration_min
    df.drop(columns=['Duration_hour','Duration_min','Arrival_Time'],inplace=True)

    ### 3. Airline 전처리
    air_count = df.Airline.value_counts().index
    airlist = [l for l in air_count if list(df.Airline).count(l) < 200]
    df.Airline = df.Airline.replace(airlist, 'Others')

    for t in range(len(air_count)):
        df.loc[df.Airline == air_count[t], 'Air_col'] = t
    df.drop(columns=['Airline'],inplace=True)

    ### 4. Additional_Info 전처리
    add_count = df.Additional_Info.value_counts().index
    additional_thing = [l for l in add_count if list(df.Additional_Info).count(l) < 20]
    df.Additional_Info = df.Additional_Info.replace(additional_thing, 'Others')

    add_count = df.Additional_Info.value_counts().index
    for t in range(len(add_count)):
        df.loc[df.Additional_Info == add_count[t], 'Add_col'] = t

    ### 5. Total_Stops 전처리
    df.loc[df.Total_Stops.isna(),'Total_Stops'] = '1 stop'

    def handle_stops(x):
        if x == 'non-stop': return 0
        return int(x.split()[0])

    df.Total_Stops = df.Total_Stops.apply(handle_stops)

    ### 6. Date_of_Journey 전처리
    df['Date_of_journey_DT'] = pd.to_datetime(df['Date_of_Journey'])
    df['weekday'] = pd.to_datetime(df['Date_of_journey_DT']).dt.weekday
    df['weekday_name'] = pd.to_datetime(df['Date_of_journey_DT']).dt.day_name()

    ### 7. Dep_Time 데이터 전처리
    df.Dep_Time = df.Dep_Time.astype(str)
    df['Dep_hour'] = df.Dep_Time.str.extract('([0-9]+)\:')
    df.drop(columns=['Dep_Time'],inplace=True)

    ### 8. 불필요 컬럼 drop
    df.drop(columns=['Date_of_Journey',
                     'Source','Destination',
                     'Date_of_journey_DT',
                     'Additional_Info',
                     'weekday'],inplace=True)

    ### 9.범주형 변수 처리
    df = pd.get_dummies(df, columns=['weekday_name','Add_col','Air_col'],drop_first=True)
    
    return df
########### function ###########
        
    
########### session ###########

if 'chk_balloon' not in st.session_state:
    st.session_state['chk_balloon'] = False

if 'chk_strline' not in st.session_state:
    st.session_state['chk_strline'] = ''

if 'choice' not in st.session_state:
    st.session_state['choice'] = ''

if 'file_name' not in st.session_state:
    st.session_state['file_name'] = ''
    
########### session ###########
       

########### define ###########

file_name = 'Data_Train.csv'
url = f'https://raw.githubusercontent.com/skfkeh/newthing/main/{file_name}'

########### define ###########

################################
#####       UI Start       #####
################################


options = st.sidebar.radio('Why is my airfare expensive?!', options=['01. Home','02. 데이터 전처리 과정','03. 시각화(plotly)'])
Home = st.container()
data_preprocessing = st.container()
visualize = st.container()
# if uploaded_file:
#    df = pd.read_excel(url)

if options == '01. Home':
    with Home:

        st.title('내 항공료는 왜 비싼 것인가💲')
        st.header('다음 항목은 사이드 메뉴를 확인해 주세요.')

        jpg_url = "https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/00f3d481-97e5-4de9-bcf2-48c82b265793/d7uteu8-e50dde9e-b8af-4fea-ab31-b7748470dc8b.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzAwZjNkNDgxLTk3ZTUtNGRlOS1iY2YyLTQ4YzgyYjI2NTc5M1wvZDd1dGV1OC1lNTBkZGU5ZS1iOGFmLTRmZWEtYWIzMS1iNzc0ODQ3MGRjOGIuanBnIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.X7DaOWcJkNe2H8jjTNtybdRCV9p5u4H_yFaOk7kMbFg"
        # st.set_page_config(layout="wide")
        st.image(jpg_url, caption="Why So Serious??!")

        st.write(f"사용한 데이터 URL : {url}")

elif options == '02. 데이터 전처리 과정':
    with data_preprocessing:
        st.image('https://www.rd.com/wp-content/uploads/2022/04/GettyImages-1140602972-e1651249657746.jpg')
        df = pd.read_csv(url)
        
        st.write("1. 확인을 위한 df.head()")
        st.dataframe(df.head())
        
        
        pre_data = preprocessing(df)
        
elif options == '03. 시각화(plotly)':
    with visualize :

        st.write("분석 알고리즘을 골라주세요")

        tab_De, tab_RF, tab_XGB = st.tabs(["DecisionTree", "RandomForest", "XGBoost"])

        #### Tab1
        with tab_De:
            col1, col2 = st.columns(2)

            st.header("Logistic")
            st.image("https://github.com/skfkeh/newthing/blob/main/img/Patrick.jpeg?raw=true", width=200)

            ts_number = col1.slider(label="test_size를 설정해주세요",
                                    min_value=0.00, max_value=1.00,
                                    step=0.10, format="%f")

            rs_number = col2.slider(label="random_state 설정",
                                        min_value=0, max_value=200,
                                    step=50, format="%d")

       # st.write(f'Test_size : {ts_number}      Random_state : {rs_text}{rs_number}')

    #### Tab2
    with tab_RF:
       st.header("RandomForest")
    #    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
       data_path = f"{os.path.dirname(os.path.abspath(__file__))}/data.csv"
       data = pd.read_csv(data_path)
       df = pd.DataFrame(data)
       df.drop('Unnamed: 0',axis=1,inplace=True)
        # 데이터 전처리
       X = df.drop('Price', axis=1)
       y = df.Price
       X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=100)

        # score 와 mse 비교
       model_pkl_path = f"{os.path.dirname(os.path.abspath(__file__))}/randomforest.pkl"
       model = joblib.load(model_pkl_path)
       st.subheader('Score 비교')
       Score_Button = st.button('예측')
       if Score_Button:
        st.write(model.score(X_train, y_train),model.score(X_test, y_test))
       train_pred = model.predict(X_train) 
       test_pred = model.predict(X_test)

       # 훈련 모델 시각화
       r1_col, r2_col = st.columns(2)
       st.subheader('모델 훈련이 잘 되었는지 시각화')
       r1_col = st.image('data:image/png;base64,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',caption='초기 파라미터 값 이미지')
    #    plt.figure(figsize=(5,5))
       plt.scatter(y_train,train_pred,marker='x')
       plt.scatter(y_test, test_pred,marker='o')
       r2_col = st.pyplot(plt)
       st.subheader('RMSE 비교') 
       train_relation_square = mean_squared_error(y_train, train_pred, squared=False)
       test_relation_square = mean_squared_error(y_test, test_pred) ** 0.5
       st.write(f' train 결정계수 : {train_relation_square}, test 결정계수 : {test_relation_square}')

       st.subheader('시각화 부분')
       SearchBtn2 = st.button('Search')

       if SearchBtn2:
           # 시각화 해보기
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=y_train,y=y_test, mode='markers',name='Train'))
        fig.add_trace(go.Scatter(x=y_test,y=test_pred,mode='markers',
                        name='Test')) # mode='lines+markers'
        fig.update_layout(title='<b>actual과 predict 비교')
        st.plotly_chart(fig)
       

    #### Tab3
    with tab_XGB:
       st.header("XGBoost")
       st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

        