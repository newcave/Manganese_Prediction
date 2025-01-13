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

def AL_RandomForest(trainX, trainY, testX, testY, n_estimators=400, max_depth=None, random_state=42):
    rf_clf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    rf_clf.fit(trainX, trainY)
    y_pred2 = rf_clf.predict(testX)
    return y_pred2

def AL_GradientBoosting(trainX, trainY, testX, testY,
                        n_estimators=200, learning_rate=0.04, max_depth=2, random_state=42):
    # (추가) 컬럼이 object일 경우 숫자형으로 한정
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns  = pd.RangeIndex(testX.shape[1])

    gbr_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    gbr_model.fit(trainX, trainY)
    y_pred2 = gbr_model.predict(testX)
    return y_pred2

def AL_XGBoosting(trainX, trainY, testX, testY,
                  n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42):
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns  = pd.RangeIndex(testX.shape[1])

    xgb_model = xgboost.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )
    xgb_model.fit(trainX, trainY)
    y_pred2 = xgb_model.predict(testX)
    return y_pred2

def AL_SVR(trainX, trainY, testX, testY):
    trainX.columns = pd.RangeIndex(trainX.shape[1])
    testX.columns  = pd.RangeIndex(testX.shape[1])

    sv_regressor = SVR(kernel='linear', C=3, epsilon=0.03)
    sv_regressor.fit(trainX, trainY)
    y_pred2 = sv_regressor.predict(testX)
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
    load_data = st.checkbox("Dam(JA) data", value=False)
    if not load_data:
        file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    else:
        file = "dataset_cleaned.csv"

    st.write("---")

# -- Main Title / Intro Section
st.title("📊 Manganese Water Quality Prediction Model")
st.markdown(
    """
    On this page, we use models such as RandomForest, GradientBoosting, and XGBoost to predict the concentration of Manganese (Mn).
    Please follow the steps below in order:
    저수지 수질을 예측하기 위해 머신러닝(RandomForest, GradientBoosting, XGBoost) 모델 적용 
    1. Upload your dataset or select the sample data  
    2. Specify the date column and the target variable (Manganese concentration)  
    3. Choose an analysis model and view the results (hyperparameter tuning is also available)
    ---
    """
)

# -- (1) 토글로 "mangan_intro.jpg" 이미지를 보여줄지 말지 결정
show_main_image = st.checkbox("Show *Introduction*", value=True)
if show_main_image:
    st.image(im2, use_column_width=True)

# -- session_state에 시뮬레이션 결과를 저장하기 위한 공간 준비
if "simulation_results" not in st.session_state:
    st.session_state["simulation_results"] = []

# -- 파일이 준비됐을 경우 로직 실행
if file:
    try:
        data = pd.read_csv(file)
    except Exception as e:
        st.warning("CSV 파일을 로드하는 동안 오류가 발생했습니다. 다시 시도해주세요.")
        st.stop()

    # -- (2) 데이터 미리보기
    with st.expander("🔍 데이터 미리보기", expanded=False):
        st.dataframe(data.head(5), use_container_width=True)

    st.write("---")

    # -- Column Selection
    with st.expander("1) 날짜 컬럼 및 망간(종속변수) 설정", expanded=True):
        columns_list = st.multiselect(
            "사용할 컬럼들을 선택해주세요",
            data.columns.tolist(),
            default=data.columns.tolist()
        )

        if len(columns_list) < 2:
            st.warning("독립변수와 종속변수를 모두 포함하도록 최소 2개 이상의 컬럼을 선택해주세요.")
        else:
            date_col = st.selectbox(
                "날짜 컬럼을 선택하세요",
                options=columns_list
            )
            y_var = st.selectbox(
                "종속변수(망간 농도) 컬럼을 선택하세요",
                options=[col for col in columns_list if col != date_col],
                index=0
            )

    st.write("---")

    if 'date_col' in locals() and 'y_var' in locals() and date_col and y_var:
        # 2) 데이터 전처리
        with st.expander("2) 🔍 데이터 전처리(Scaling, 날짜 처리)", expanded=True):
            st.write("✅ 선택된 날짜 컬럼: ", date_col)
            st.write("✅ 선택된 종속변수(망간 농도): ", y_var)

            data = data.rename(columns={date_col: "set_date"})
            data['set_date'] = pd.to_datetime(data['set_date'], errors='coerce')
            data['month'] = data['set_date'].dt.month.astype(float)
            data = data.dropna(subset=["set_date", y_var])

            numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
            if 'set_date' in numeric_cols:
                numeric_cols.remove('set_date')

            scaler = MinMaxScaler()
            scaled_data = data.copy()
            scaled_data[numeric_cols] = scaler.fit_transform(scaled_data[numeric_cols])

            st.markdown("**전처리된 데이터 미리보기**")
            st.dataframe(scaled_data.head(5), use_container_width=True)

        # -- 최종 데이터에서 날짜 컬럼 제외
        final_data = scaled_data.drop(['set_date'], axis=1)
        X = final_data.drop([y_var], axis=1)
        y = final_data[y_var]

        # -- Train/Test Split
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

        ########################################################################
        # (중요) NaN / Inf 제거 & 숫자형 칼럼만 사용
        ########################################################################
        # 1) 숫자형 컬럼만 사용
        trainX = trainX.select_dtypes(include=[np.number])
        testX  = testX.select_dtypes(include=[np.number])

        # 2) Inf -> NaN 변환
        trainX = trainX.replace([np.inf, -np.inf], np.nan)
        testX  = testX.replace([np.inf, -np.inf], np.nan)

        # 3) NaN이 있는 행 제거 (X, Y 동기화)
        #    trainX가 NaN인 행 => trainY에서도 같은 index 제거
        before_train_size = len(trainX)
        train_mask = trainX.notnull().all(axis=1)  # 모든 칼럼이 notnull인 행만 True
        trainX = trainX[train_mask]
        trainY = trainY[train_mask]
        after_train_size = len(trainX)

        # test도 동일하게
        before_test_size = len(testX)
        test_mask = testX.notnull().all(axis=1)
        testX = testX[test_mask]
        testY = testY[test_mask]
        after_test_size = len(testX)

        # -- 로그로 확인해볼 수도 있음
        st.write(f"Train Data: {before_train_size} -> {after_train_size} (rows after drop NaN/Inf)")
        st.write(f"Test  Data: {before_test_size} -> {after_test_size} (rows after drop NaN/Inf)")

        ########################################################################
        # 이후 모델 학습
        ########################################################################
        model_list = ["Random Forest", "Gradient Boosting", "XGBoost"]
        performance_list = ["RMSE", "R2", "MSE", "MAE"]

        with st.expander("3) 모델 선택 및 예측 (하이퍼파라미터 조정)", expanded=True):
            selected_model = st.selectbox("예측 모델을 선택하세요", model_list)

            st.markdown("**하이퍼파라미터 설정**")
            if selected_model == "Random Forest":
                n_estimators = st.number_input("n_estimators (트리 개수)", min_value=50, max_value=2000, value=300, step=50)
                max_depth = st.number_input("max_depth (트리 깊이)", min_value=1, max_value=100, value=5, step=1)
                param_dict = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }
            elif selected_model == "Gradient Boosting":
                n_estimators = st.number_input("n_estimators (트리 개수)", min_value=50, max_value=2000, value=300, step=50)
                learning_rate = st.slider("learning_rate", min_value=0.001, max_value=1.0, value=0.04, step=0.01)
                max_depth = st.number_input("max_depth (트리 깊이)", min_value=1, max_value=50, value=2, step=1)
                param_dict = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth
                }
            elif selected_model == "XGBoost":
                n_estimators = st.number_input("n_estimators (트리 개수)", min_value=50, max_value=2000, value=500, step=50)
                learning_rate = st.slider("learning_rate", min_value=0.001, max_value=1.0, value=0.05, step=0.01)
                max_depth = st.number_input("max_depth (트리 깊이)", min_value=1, max_value=50, value=3, step=1)
                param_dict = {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth
                }
            else:
                param_dict = {}

            if st.button("모델 훈련 및 예측하기"):
                if selected_model == "Random Forest":
                    yhat = AL_RandomForest(trainX, trainY, testX, testY,
                                           n_estimators=param_dict["n_estimators"],
                                           max_depth=param_dict["max_depth"])
                elif selected_model == "Gradient Boosting":
                    yhat = AL_GradientBoosting(trainX, trainY, testX, testY,
                                               n_estimators=param_dict["n_estimators"],
                                               learning_rate=param_dict["learning_rate"],
                                               max_depth=param_dict["max_depth"])
                elif selected_model == "XGBoost":
                    yhat = AL_XGBoosting(trainX, trainY, testX, testY,
                                         n_estimators=param_dict["n_estimators"],
                                         learning_rate=param_dict["learning_rate"],
                                         max_depth=param_dict["max_depth"])
                else:
                    st.warning("모델이 선택되지 않았습니다.")
                    st.stop()

                st.session_state["simulation_results"].append(
                    (selected_model, testY.values, yhat, param_dict)
                )
                st.success(f"✅ 예측 완료! 모델: {selected_model}")

    else:
        st.warning("날짜 컬럼과 종속변수(망간) 컬럼을 올바르게 선택해주세요.")

# -- 시뮬레이션 결과 표시
if len(st.session_state["simulation_results"]) > 0:
    st.write("---")
    st.subheader("📈 시뮬레이션 결과(누적)")

    total_sims = len(st.session_state["simulation_results"])
    for idx, (model_name, actual_vals, pred_vals, used_params) in enumerate(reversed(st.session_state["simulation_results"]), start=1):
        sim_number = total_sims - idx + 1  
        st.markdown(f"### ▶ 시뮬레이션 #{sim_number} (모델: {model_name})")
        if used_params:
            st.write("**사용한 하이퍼파라미터:**", used_params)

        performance_list = ["RMSE", "R2", "MSE", "MAE"]
        cols = st.columns(4)
        for i, pi in enumerate(performance_list):
            score = Performance_index(actual_vals, pred_vals, pi)
            cols[i].metric(label=pi, value=f"{score:.4f}")
        
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
            xaxis_title='Data Index(#)',
            yaxis_title='Concentration(mg/L)',
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
