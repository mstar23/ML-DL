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
       st.subheader('모델 훈련이 잘 되었는지 시각화')
    #    plt.figure(figsize=(5,5))
       plt.scatter(y_train,train_pred,marker='x')
       plt.scatter(y_test, test_pred,marker='o')
       st.pyplot(plt)
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

        