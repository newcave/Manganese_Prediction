import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost
import streamlit as st
from PIL import Image
import plotly.graph_objs as go


def AL_RandomForest(trainX, trainY, testX, testY):
    rf_clf = RandomForestRegressor(n_estimators=500)
    rf_clf.fit(trainX, trainY)

    y_pred2 = rf_clf.predict(testX)

    return y_pred2

def AL_GradientBoosting(trainX, trainY, testX, testY):
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate = 0.05)
    gbr_model.fit(trainX, trainY)

    y_pred2 = gbr_model.predict(testX)

    return y_pred2

def AL_SVR(trainX, trainY, testX, testY):

    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    sv_regressor = SVR(kernel='linear', C=3, epsilon=0.03)
    sv_regressor.fit(trainX, trainY)

    y_pred2 = sv_regressor.predict(testX)

    return y_pred2


def Performance_index(obs, pre, mod_str):
    if mod_str == 'R2':
        pf_index = r2_score(obs, pre)
    elif mod_str == 'RMSE':
        s1 = mean_squared_error(obs, pre)
        pf_index = np.sqrt(s1)
    elif mod_str == 'MSE':
        pf_index = mean_squared_error(obs, pre)
    elif mod_str == 'MAE':
        pf_index = mean_absolute_error(obs, pre)

    return pf_index

def AL_XGBoosting(trainX, trainY, testX, testY):
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = xgboost.XGBRegressor(n_estimators=500, learning_rate = 0.05, max_depth=7)
    gbr_model.fit(trainX, trainY)

    y_pred2 = gbr_model.predict(testX)

    return y_pred2


im = Image.open("AI_Lab_logo.jpg")
im2 = Image.open("mangan_intro.jpg")
st.set_page_config(
    page_title="망간 수질 예측(Manganese water quality prediction model)",
    page_icon='📈',
    layout="wide",
)


with st.sidebar :

    st.image(im, width = 100)

    st.header('🐳 망간 수질예측')
    st.subheader('', divider='grey')


    load_data = st.checkbox("Sample 데이터(주암댐)", value=False)

    # 데이터 로드 여부에 따라 실행
    if not load_data:
        file = st.file_uploader("  Upload .CSV 파일", type=["csv"])
    else:
        # 데이터 로드하지 않을 경우 안내 메시지 출력
        # st.write("기본 예제 데이터(주암댐)")
        file = "dataset_cleaned.csv"

    st.subheader('', divider='rainbow')
    
placeholder = st.image(im2, width = 100, use_column_width="auto")

process_val = 0 #진행 변수추가

if load_data :
    placeholder.empty()
    data = pd.read_csv(file)

    # 데이터 열 선택
    columns_list = st.multiselect("🐟 독립변수와 종속변수(망간 데이터)를 선택", data.columns, placeholder="컬럼에서 독립변수와 종속변수(망간 데이터)를 모두 선택해주세요.")
    st.dataframe(data.iloc[0:3][columns_list], width=1000, height=150)
    
    set_date =""
    if columns_list:
        
        try:
            set_date = st.selectbox('🐟 날짜 컬럼 선택', columns_list, placeholder ="날짜(DATE) 컬럼을 선택해주세요.", index=None)
            data = data.rename(columns={set_date: "set_date"})
            data['set_date'] = pd.to_datetime(data['set_date'])
            data['month'] = data['set_date'].dt.month.astype(float)
            # st.dataframe(data, width=1000, height=150)
            process_val+=1
        
        except:
            # st.warning('날짜 컬럼을 선택해주세요')
            st.write("")
     
        # 종속 데이터 선택
        if process_val==1:
            try:
                y_var = st.selectbox('🐟 종속변수(망간 데이터 칼럼) 선택', columns_list, placeholder ="망간 데이터 칼럼을 선택해주세요.",index=None)
                data = data.dropna()

                scaler = MinMaxScaler()
                _train_data = data.drop(['set_date'], axis=1)
                scaler.fit(_train_data)
                train_data_ = scaler.transform(_train_data)
                train_data = pd.DataFrame(data, columns=data.columns)
                train_data[_train_data.columns] = train_data_

                st.dataframe(data, width=1000, height=150)
                process_val+=1
                
                # set_date 삭제
                train_data.drop(['set_date'], axis=1, inplace=True)

                # 환경 변수
                model_list = ["Gradient Boosting", "Random Forest", "XGBoost"]  # 분석 모델 리스트 설정 : LSTM, GBM, RF, SVR
                performance_list = ["RMSE", "R2", "MSE", "MAE"]  # 분석 성능평가 리스트 설정 : RMSE, R2, MSE, MAE

                # selected_model = st.sidebar.radio('Select an option:', model_list)
                st.subheader('', divider='rainbow')
                selected_model = st.sidebar.radio('📈 모델을 선택해 주세요', model_list)
                
                # 데이터 분할
                X = train_data.drop([y_var], axis=1)
                y = train_data[y_var]
                trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)


                # 모델 학습
                if selected_model == "Gradient Boosting":
                    predict = AL_GradientBoosting(trainX, trainY, testX, testY)
                elif selected_model == "Random Forest":
                    predict = AL_RandomForest(trainX, trainY, testX, testY)
                elif selected_model == "SVR":
                    predict = AL_SVR(trainX, trainY, testX, testY)
                elif selected_model == "XGBoost":
                    predict = AL_XGBoosting(trainX, trainY, testX, testY)

                # 예측
                yhat = predict
                actual = testY

                # 정확도 출력
                # 성과지표 표출 부분 : 적용 항목은 confing > performance_list[] 참조
                
                st.sidebar.subheader('', divider='rainbow')
                st.sidebar.markdown(f'#### 🌵 {selected_model}')

                for pi in performance_list:
                    rmse = Performance_index(actual, yhat, pi)
                    formatted_pi = f'{pi:<6}'
                    rmse_formatted = f'{rmse:.3f}'
                    st.sidebar.write(f'🌱 {formatted_pi} : {rmse_formatted}')
                    
                st.markdown("🐟 망간 수질 실제 VS 예측 그래프")
                st.markdown(f'#### 🌵 {selected_model}')
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=np.arange(len(actual)), y=actual, mode='lines+markers', name='Actual', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=np.arange(len(yhat)), y=yhat, mode='lines+markers', name='Predicted', line=dict(color='red')))

                fig.update_layout(
                    xaxis_title='Samples',
                    yaxis_title='Mn(%)',
                    legend=dict(orientation='h', y=1.1),
                    autosize=True,
                    width=1200,
                    height=600,
                    margin=dict(l=10, r=10, t=10, b=10),  
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(showline=True, linewidth=2, linecolor='black'),  # X-axis border
                    yaxis=dict(showline=True, linewidth=2, linecolor='black')  # Y-axis border
                )
                st.plotly_chart(fig)
            except:
                # st.warning("종속변수를 선택해야 넘어갑니다") 
                st.write("")
