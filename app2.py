import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost
from PIL import Image
import plotly.graph_objs as go

##########################
# 1. Utility Functions   #
##########################

def AL_RandomForest(trainX, trainY, testX, testY):
    rf_clf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_clf.fit(trainX, trainY)
    y_pred2 = rf_clf.predict(testX)
    return y_pred2

def AL_GradientBoosting(trainX, trainY, testX, testY):
    # Temporarily rename columns as integers to avoid fitting errors
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    gbr_model.fit(trainX, trainY)
    y_pred2 = gbr_model.predict(testX)
    return y_pred2

def AL_SVR(trainX, trainY, testX, testY):
    # Temporarily rename columns as integers to avoid fitting errors
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    sv_regressor = SVR(kernel='linear', C=3, epsilon=0.03)
    sv_regressor.fit(trainX, trainY)
    y_pred2 = sv_regressor.predict(testX)
    return y_pred2

def AL_XGBoosting(trainX, trainY, testX, testY):
    # Temporarily rename columns as integers to avoid fitting errors
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns = pd.RangeIndex(testX.shape[1])

    gbr_model = xgboost.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
    gbr_model.fit(trainX, trainY)
    y_pred2 = gbr_model.predict(testX)
    return y_pred2

def Performance_index(obs, pre, mod_str):
    if mod_str == 'R2':
        pf_index = r2_score(obs, pre)
    elif mod_str == 'RMSE':
        pf_index = np.sqrt(mean_squared_error(obs, pre))
    elif mod_str == 'MSE':
        pf_index = mean_squared_error(obs, pre)
    elif mod_str == 'MAE':
        pf_index = mean_absolute_error(obs, pre)
    return pf_index

##########################
# 2. Streamlit App       #
##########################

# -- Images/Logos
im = Image.open("AI_Lab_logo.jpg")
im2 = Image.open("mangan_intro.jpg")

# -- Page Config
st.set_page_config(
    page_title="망간 수질 예측(Manganese Prediction)",
    page_icon="📈",
    layout="wide"
)

# -- Sidebar Header
with st.sidebar:
    st.image(im, width=150)
    st.title("망간 수질예측")
    st.write("---")

    # Checkbox for sample data
    load_data = st.checkbox("샘플 데이터 사용 (주암댐)", value=False)
    if not load_data:
        file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    else:
        # When checked, use default dataset file name
        file = "dataset_cleaned.csv"

    st.write("---")

# -- Main Title / Intro Section
st.title("📊 망간 수질 예측 모델")
st.markdown(
    """
    이 페이지에서는 망간(Mn) 농도를 예측하기 위해 **RandomForest, GradientBoosting, XGBoost** 등의 
    모델을 활용합니다. 아래의 단계를 차례대로 진행해주세요:
    1. **데이터 업로드** 또는 **샘플 데이터 선택**  
    2. **날짜 컬럼, 종속변수(망간 농도) 설정**  
    3. **분석 모델 선택 및 결과 확인**  
    ---
    """
)

# -- Image Placeholder
placeholder = st.empty()
if file == "dataset_cleaned.csv":
    placeholder.image(im2, use_column_width=True)

# -- session_state에 시뮬레이션 결과를 저장하기 위한 공간 준비
if "simulation_results" not in st.session_state:
    # simulation_results 리스트 내부에 (모델 이름, 실제값, 예측값) 튜플을 저장할 예정
    st.session_state["simulation_results"] = []

# -- If data file is provided or sample is chosen
if file:
    try:
        data = pd.read_csv(file)
    except Exception as e:
        st.warning("CSV 파일을 로드하는 동안 오류가 발생했습니다. 다시 시도해주세요.")
        st.stop()

    with st.expander("🔍 데이터 미리보기", expanded=False):
        st.dataframe(data.head(5), use_container_width=True)

    st.write("---")

    # -- Column Selection Expander
    with st.expander("1) 날짜 컬럼 및 망간(종속변수) 설정", expanded=True):
        # Multi-select for columns that might be used
        columns_list = st.multiselect(
            "사용할 컬럼들을 선택해주세요",
            data.columns.tolist(),
            default=data.columns.tolist()
        )

        # We want at least something selected
        if len(columns_list) < 2:
            st.warning("독립변수와 종속변수를 모두 포함하도록 최소 2개 이상의 컬럼을 선택해주세요.")
        else:
            # Select date column
            date_col = st.selectbox(
                "날짜 컬럼을 선택하세요",
                options=columns_list
            )

            # Select target column (Manganese)
            y_var = st.selectbox(
                "종속변수(망간 농도) 컬럼을 선택하세요",
                options=[col for col in columns_list if col != date_col],
                index=0
            )

    st.write("---")

    # -- Check if user has selected columns properly
    if date_col and y_var and (date_col in columns_list) and (y_var in columns_list):
        # Data Preprocessing
        with st.expander("2) 데이터 전처리(Scaling, 날짜 처리)", expanded=True):
            st.write("✅ 선택된 날짜 컬럼: ", date_col)
            st.write("✅ 선택된 종속변수(망간 농도): ", y_var)

            # Rename, handle date
            data = data.rename(columns={date_col: "set_date"})
            data['set_date'] = pd.to_datetime(data['set_date'], errors='coerce')
            # Extract month as a float feature
            data['month'] = data['set_date'].dt.month.astype(float)
            # Remove rows with missing values
            data = data.dropna(subset=["set_date", y_var])  

            # Scale all numeric columns except 'set_date'
            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if 'set_date' in numeric_cols:
                numeric_cols.remove('set_date')

            # MinMaxScaler
            scaler = MinMaxScaler()
            scaled_data = data.copy()
            scaled_data[numeric_cols] = scaler.fit_transform(scaled_data[numeric_cols])

            st.markdown("**전처리된 데이터 미리보기**")
            st.dataframe(scaled_data.head(5), use_container_width=True)

        # Prepare data for modeling
        # Drop date column from final training
        final_data = scaled_data.drop(['set_date'], axis=1)
        X = final_data.drop([y_var], axis=1)
        y = final_data[y_var]

        # Train/Test Split
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

        # -- Model Selection
        model_list = ["Random Forest", "Gradient Boosting", "XGBoost"]
        performance_list = ["RMSE", "R2", "MSE", "MAE"]

        with st.expander("3) 모델 선택 및 예측", expanded=True):
            selected_model = st.selectbox("예측 모델을 선택하세요", model_list)

            # Run training only when the user clicks this button
            if st.button("모델 훈련 및 예측하기"):
                # 모델에 따라 시뮬레이션 실행
                if selected_model == "Gradient Boosting":
                    yhat = AL_GradientBoosting(trainX, trainY, testX, testY)
                elif selected_model == "Random Forest":
                    yhat = AL_RandomForest(trainX, trainY, testX, testY)
                elif selected_model == "XGBoost":
                    yhat = AL_XGBoosting(trainX, trainY, testX, testY)
                else:
                    st.warning("모델이 선택되지 않았습니다.")
                    st.stop()

                # 예측이 완료되면 session_state에 결과 저장
                # (모델 이름, 실제값, 예측값)을 튜플로 저장
                st.session_state["simulation_results"].append((selected_model, testY.values, yhat))

                st.success(f"✅ 예측 완료! 모델: {selected_model}")

    else:
        st.warning("날짜 컬럼과 종속변수(망간) 컬럼을 올바르게 선택해주세요.")

# -- 이제 session_state에 누적된 시뮬레이션 결과를 보여줍니다.
# -- Footer 위에 추가로 보여줄 수 있습니다.
if len(st.session_state["simulation_results"]) > 0:
    st.write("---")
    st.subheader("📈 시뮬레이션 결과(누적)")
    
    # 여러 번 돌린 시뮬레이션 결과를 순차적으로 표시
    for idx, (model_name, actual_vals, pred_vals) in enumerate(st.session_state["simulation_results"], start=1):
        st.markdown(f"### ▶ 시뮬레이션 #{idx} (모델: {model_name})")

        # 성능 지표 계산
        performance_list = ["RMSE", "R2", "MSE", "MAE"]
        # 성능 지표 출력
        cols = st.columns(4)
        for i, pi in enumerate(performance_list):
            score = Performance_index(actual_vals, pred_vals, pi)
            cols[i].metric(label=pi, value=f"{score:.4f}")
        
        # 그래프 시각화
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(actual_vals)),
                y=actual_vals,
                mode='lines+markers',
                name='Actual',
                line=dict(color='green')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(pred_vals)),
                y=pred_vals,
                mode='lines+markers',
                name='Predicted',
                line=dict(color='red')
            )
        )
        fig.update_layout(
            xaxis_title='테스트 데이터 인덱스',
            yaxis_title='망간 농도(Mn)',
            legend=dict(orientation='h', y=1.1),
            autosize=True,
            width=1200,
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showline=True, linewidth=2, linecolor='black'),
            yaxis=dict(showline=True, linewidth=2, linecolor='black')
        )
        st.plotly_chart(fig, use_container_width=True)

# -- Footer / Credits
st.write("---")
st.markdown(
    """
    <p style='text-align: center; color: grey;'>
      ⓒ 2025 K-water AI Lab. All rights reserved.  
      | 문의: <a href='mailto:sunghoonkim@kwater.or.kr'>sunghoonkim@kwater.or.kr</a>
    </p>
    """,
    unsafe_allow_html=True
)
