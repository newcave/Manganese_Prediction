import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from PIL import Image

# 페이지 설정
st.set_page_config(
    page_title="6-Step Process Flow Chart",
    page_icon="🔄",
    layout="wide"
)

# Session State 초기화
if 'selected_process' not in st.session_state:
    st.session_state.selected_process = "1️⃣ Raw Water Quality Prediction"

if 'redirected' not in st.session_state:
    st.session_state.redirected = False

# 프로세스 레이블 업데이트
process_labels = {
    "A": "Raw Water Quality Prediction",
    "B": "Coagulation/Flocculation",
    "C": "Filtration",
    "D": "Disinfection",
    "E": "DPBs",
    "F": "Water Demand"
}

# 프로세스 설명 업데이트
process_descriptions = {
    "1️⃣ Raw Water Quality Prediction": "**Raw Water Quality Prediction:** Predicting the quality of raw water before processing.",
    "2️⃣ Coagulation/Flocculation": "**Coagulation/Flocculation:** Combining chemicals to remove suspended solids from water.",
    "3️⃣ Filtration": "**Filtration:** Filtering out remaining particles from water.",
    "4️⃣ Disinfection": "**Disinfection:** Eliminating pathogens to ensure water safety.",
    "5️⃣ DPBs": "**DPBs:** Managing Deposits, Pitting, and Corrosion in water systems.",
    "6️⃣ Water Demand": "**Water Demand:** Assessing and meeting the water demand requirements.",
    "A": "**Raw Water Quality Prediction:** Predicting the quality of raw water before processing.",
    "B": "**Coagulation/Flocculation:** Combining chemicals to remove suspended solids from water.",
    "C": "**Filtration:** Filtering out remaining particles from water.",
    "D": "**Disinfection:** Eliminating pathogens to ensure water safety.",
    "E": "**DPBs:** Managing Deposits, Pitting, and Corrosion in water systems.",
    "F": "**Water Demand:** Assessing and meeting the water demand requirements."
}

# 프로세스 링크 업데이트
process_links = {
    "1️⃣ Raw Water Quality Prediction": "https://mn-prediction-kwaterailab.streamlit.app/",
    "3️⃣ Filtration": "https://newcave-230413-mebrane-sem-image-analysis-010-app-sem-tss0v2.streamlit.app/"  # Updated Filtration app URL
}

# 프로세스 A 또는 C 선택 시 리디렉션
if st.session_state.selected_process in process_links and not st.session_state.redirected:
    components.html(
        f"""
        <script>
            window.location.href = "{process_links[st.session_state.selected_process]}";
        </script>
        """,
        height=0,
        width=0
    )
    st.session_state.redirected = True

# 함수: 노드 색상 및 테두리 업데이트
def get_nodes(selected):
    try:
        # 첫 번째 단어에서 숫자 추출 (예: "1️⃣"에서 "1" 추출)
        number_str = selected.split()[0][0]  # '1'
        number = int(number_str)
        selected_id = chr(64 + number)  # 1 -> 'A', 2 -> 'B', ..., 6 -> 'F'
    except (IndexError, ValueError):
        selected_id = 'A'  # 기본값 설정 (필요에 따라 변경 가능)
    
    nodes = []
    for node_id in ["A", "B", "C", "D", "E", "F"]:
        if node_id == selected_id:
            # 선택된 노드: 배경색 주황색 및 테두리 색상 진하게 변경
            node_color = {
                "background": "#FFA500",  # 주황색
                "border": "#FF8C00",
                "highlight": {
                    "background": "#FFB347",
                    "border": "#FF8C00"
                }
            }
        else:
            # 기본 노드 색상
            node_color = {
                "background": "#ADD8E6",  # 연한 파랑
                "border": "#000000",
                "highlight": {
                    "background": "#87CEFA",
                    "border": "#000000"
                }
            }
        
        nodes.append(
            Node(
                id=node_id,
                label=f"Process {node_id}\n({process_labels[node_id]})",
                size=32,
                color=node_color
            )
        )
    return nodes

# 함수: 엣지 정의
def get_edges():
    edges = [
        Edge(source="A", target="B", label="→"),
        Edge(source="B", target="C", label="→"),
        Edge(source="C", target="D", label="↓"),
        Edge(source="D", target="E", label="→"),
        Edge(source="E", target="F", label="→"),
        Edge(source="F", target="A", label="↑"),
    ]
    return edges

# 그래프 설정
def get_config():
    config = Config(
        height=600,
        width=800,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        node={'color': '#ADD8E6'},
        link={'color': '#808080', 'labelHighlightBold': True},
    )
    return config

# 함수: 랜덤 시계열 데이터 생성 및 세션 상태에 저장
def get_timeseries_data(process_name, points=50):
    if 'timeseries_data' not in st.session_state:
        st.session_state.timeseries_data = {}
    
    if process_name not in st.session_state.timeseries_data:
        # 랜덤 시드 고정 (일관된 데이터 생성)
        seed = hash(process_name) % (2**32)
        np.random.seed(seed)
        dates = pd.date_range(start='2023-01-01', periods=points)
        values = np.random.randn(points).cumsum()  # 랜덤 누적 합

        df = pd.DataFrame({
            'Date': dates,
            'Value': values
        })
        st.session_state.timeseries_data[process_name] = df
    
    return st.session_state.timeseries_data[process_name]

# Plotly 시계열 차트 생성 함수
def create_timeseries_chart(df, process_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Value'],
        mode='lines+markers',
        line=dict(color='royalblue'),
        marker=dict(size=4),
        name=f'{process_name} Time Series'
    ))

    fig.update_layout(
        title=f"📈 {process_name} - Random Time Series Data",
        xaxis_title="Date",
        yaxis_title="Measurement Value",
        autosize=True,
        width=800,
        height=400,
        plot_bgcolor='white'
    )

    return fig

# 사이드바 설정
with st.sidebar:
    st.title("⚙️ Select Process")
    
    selected_process_sidebar = st.radio(
        "Choose a process to explore:",
        [
            "1️⃣ Raw Water Quality Prediction",
            "2️⃣ Coagulation/Flocculation",
            "3️⃣ Filtration",
            "4️⃣ Disinfection",
            "5️⃣ DPBs",
            "6️⃣ Water Demand"
        ],
        index=[
            "1️⃣ Raw Water Quality Prediction",
            "2️⃣ Coagulation/Flocculation",
            "3️⃣ Filtration",
            "4️⃣ Disinfection",
            "5️⃣ DPBs",
            "6️⃣ Water Demand"
        ].index(st.session_state.selected_process)
    )
    
    # 사이드바에서 선택된 경우, 세션 상태 업데이트
    if selected_process_sidebar != st.session_state.selected_process:
        st.session_state.selected_process = selected_process_sidebar
        st.session_state.redirected = False  # Reset redirection if process changes
    
    # Disinfection 선택 시 추가 입력 슬라이더
    if st.session_state.selected_process.startswith("4️⃣"):
        st.write("---")
        st.header("모델 인풋 설정")
        
        # 이미지 불러오기
        try:
            im = Image.open("AI_Lab_logo.jpg")
            st.image(im, caption=" ", use_column_width=True)
        except FileNotFoundError:
            st.write("Logo image not found.")
        
        # 사용자 입력
        DOC = st.slider("DOC (mg/L)", 0.0, 10.0, 5.0)
        NH3 = st.slider("Surrogate Variable (mg/L)", 0.0, 5.0, 0.5)
        Cl0 = st.slider("현재농도 Cl0 (mg/L)", 0.0, 5.0, 1.5)
        Temp = st.slider("Temperature (°C)", 0.0, 35.0, 20.0)
        max_time = st.slider("최대예측시간 (hrs)", 1, 24, 5)
        
        # 추가적인 범위
        st.header("EPA 모델 k1, k2 범위 설정")
        k1_low = st.slider("AI High1 (k1 최대 적정범위)", 0.01, 5.0, 3.5)
        k1_high = st.slider("AI Low1 (k1 최소 적정범위)", 0.01, 5.0, 2.0)
        k2_low = st.slider("AI High2 (k2 최대 적정범위)", 0.01, 5.0, 0.1)
        k2_high = st.slider("AI Low2 (k2 최소 적정범위)", 0.01, 5.0, 0.5)
        
        # 세션에 저장
        if 'disinfection_inputs' not in st.session_state:
            st.session_state.disinfection_inputs = {}
        
        st.session_state.disinfection_inputs['DOC'] = DOC
        st.session_state.disinfection_inputs['NH3'] = NH3
        st.session_state.disinfection_inputs['Cl0'] = Cl0
        st.session_state.disinfection_inputs['Temp'] = Temp
        st.session_state.disinfection_inputs['max_time'] = max_time
        st.session_state.disinfection_inputs['k1_low'] = k1_low
        st.session_state.disinfection_inputs['k1_high'] = k1_high
        st.session_state.disinfection_inputs['k2_low'] = k2_low
        st.session_state.disinfection_inputs['k2_high'] = k2_high
    
    st.write("---")
    st.info(f"🔍 **Selected Process:** {st.session_state.selected_process}")

# 메인 레이아웃
col1, col2 = st.columns([1, 1])  # Changed from [2, 1] to [1, 1] to reduce distance

with col1:
    st.subheader("🔵 Processes Configurations 🔵")
    # Flow-Chart (Agraph)
    nodes = get_nodes(st.session_state.selected_process)
    edges = get_edges()
    config = get_config()
    
    response = agraph(nodes=nodes, edges=edges, config=config)

    if response and 'clickedNodes' in response and len(response['clickedNodes']) > 0:
        clicked_node_id = response['clickedNodes'][0]['id']
        process_number = ord(clicked_node_id) - 64  # 'A' -> 1
        process_name = process_labels.get(clicked_node_id, "Unknown Process")
        st.session_state.selected_process = f"{process_number}️⃣ {process_name}"
        st.session_state.redirected = False  # Reset redirection when a node is clicked

# --------------------------------------------------------------------
# col2 영역에서 'Plotly Circle Chart' 대신 'Agraph'를 사용해 4개 노드 표시
# --------------------------------------------------------------------
with col2:
    # 노드 클릭 여부와 상관없이 process_name을 세션값 기준으로 안전하게 추출
    try:
        num_str = st.session_state.selected_process.split()[0][0]  # 예: "1️⃣" -> '1'
        num = int(num_str)
        fallback_node_id = chr(64 + num)  # 1 -> 'A'
        process_name = process_labels.get(fallback_node_id, "Unknown Process")
    except (IndexError, ValueError):
        process_name = "Unknown Process"

    # 프로세스 A 또는 C 선택 시 안내
    if st.session_state.selected_process in process_links:
        # Define descriptions for each linked process
        linked_descriptions = {
            "1️⃣ Raw Water Quality Prediction": "🔄 Manganese Prediction in reservoirs",
            "3️⃣ Filtration": "🔄 Membrane Analysis"  # Updated description for Filtration
        }
        description = linked_descriptions.get(st.session_state.selected_process, "🔄 Process Overview")
        st.info(description)
        
        # Define links for each linked process
        linked_urls = {
            "1️⃣ Raw Water Quality Prediction": process_links["1️⃣ Raw Water Quality Prediction"],
            "3️⃣ Filtration": process_links["3️⃣ Filtration"]
        }
        st.markdown(f"[👉 Click]({linked_urls[st.session_state.selected_process]})")
    else:
        st.subheader(f"** {process_name} - Key Parameters")
    
        # 4개 노드만 있는 Agraph 구성
        # (1) Manganese, (2) Algae, (3) Synedra, (4) 2-MIB
        node_list = [
            Node(id="Manganese", label="Manganese", size=30, color="#4F81BD", shape='database'),  # 파랑
            Node(id="Algae",     label="Algae",     size=30, color="#9BBB59", shape='box'),      # 연두
            Node(id="Synedra",   label="Synedra",   size=30, color="#F79646", shape='ellipse'),  # 주황
            Node(id="2-MIB",     label="2-MIB",     size=30, color="#C0504D", shape='ellipse')   # 붉은색
        ]
        # 엣지 정의: 노드 간 연결 추가
        edge_list = [
            Edge(source="Manganese", target="Algae", label="→"),
            Edge(source="Algae", target="Synedra", label="→"),
            Edge(source="Synedra", target="2-MIB", label="→"),
            Edge(source="2-MIB", target="Manganese", label="→"),
        ]

        # 두 번째 Agraph 설정
        config2 = Config(
            height=600,
            width=600,
            directed=True,  # 방향성 있는 엣지 표시
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            node={'color': '#ADD8E6'},
            link={'color': '#808080', 'labelHighlightBold': True},
        )

        # Agraph 출력
        agraph(nodes=node_list, edges=edge_list, config=config2)

# 메인 타이틀
st.title("📊 Connected Process Flow Chart & Simulator")

# Disinfection 프로세스 로직
if st.session_state.selected_process.startswith("4️⃣"):
    if 'disinfection_inputs' not in st.session_state:
        st.warning("사이드바에서 Disinfection 프로세스의 입력을 설정해 주세요.")
    else:
        inputs = st.session_state.disinfection_inputs
        DOC = inputs['DOC']
        NH3 = inputs['NH3']
        Cl0 = inputs['Cl0']
        Temp = inputs['Temp']
        max_time = inputs['max_time']
        k1_low = inputs['k1_low']
        k1_high = inputs['k1_high']
        k2_low = inputs['k2_low']
        k2_high = inputs['k2_high']
        
        # EPA 모델
        try:
            k1_EPA = np.exp(-0.442 + 0.889 * np.log(DOC) + 0.345 * np.log(7.6 * NH3) 
                            - 1.082 * np.log(Cl0) + 0.192 * np.log(Cl0 / DOC))
            k2_EPA = np.exp(-4.817 + 1.187 * np.log(DOC) + 0.102 * np.log(7.6 * NH3) 
                            - 0.821 * np.log(Cl0) - 0.271 * np.log(Cl0 / DOC))
        except:
            st.error("EPA 모델 계산을 위한 입력값이 유효하지 않습니다.")
            st.stop()
        
        # Two-phase 모델
        try:
            A_Two_phase = np.exp(
                0.168 - 0.148 * np.log(Cl0 / DOC) + 0.29 * np.log(1) - 0.41 * np.log(Cl0)
                + 0.038 * np.log(1) + 0.0554 * np.log(NH3) + 0.185 * np.log(Temp)
            )
            k1_Two_phase = np.exp(
                5.41 - 0.38 * np.log(Cl0 / DOC) + 0.274 * np.log(NH3)
                - 1.12 * np.log(Temp) + 0.05 * np.log(1) - 0.854 * np.log(7)
            )
            k2_Two_phase = np.exp(
                -7.13 + 0.864 * np.log(Cl0 / DOC) + 2.63 * np.log(DOC)
                - 2.55 * np.log(Cl0) + 0.62 * np.log(1) + 0.16 * np.log(1)
                + 0.48 * np.log(NH3) + 1.03 * np.log(Temp)
            )
        except:
            st.error("Two-phase 모델 계산을 위한 입력값이 유효하지 않습니다.")
            st.stop()
        
        # 시간
        time_range = np.linspace(0, max_time, 100)
        
        # EPA 모델 (기본)
        C_EPA = np.where(
            time_range <= 5,
            Cl0 * np.exp(-k1_EPA * time_range),
            Cl0 * np.exp(5 * (k2_EPA - k1_EPA)) * np.exp(-k2_EPA * time_range)
        )
        
        # 시간 변동(랜덤)
        def apply_time_based_variation(array, max_time):
            variation_factors = 1 + (time_range / max_time * 2) * np.random.uniform(-0.2, 0.4, size=array.shape)
            return array * variation_factors
        
        C_EPA_varied = apply_time_based_variation(C_EPA, max_time)
        
        # Two-phase 모델 (기본)
        C_Two_phase = Cl0 * (
            A_Two_phase * np.exp(-k1_Two_phase * time_range) 
            + (1 - A_Two_phase) * np.exp(-k2_Two_phase * time_range)
        )
        
        # EPA 모델 (사용자 지정 범위)
        C_EPA_low = np.where(
            time_range <= 5,
            Cl0 * np.exp(-k1_low * time_range),
            Cl0 * np.exp(5 * (k2_low - k1_low)) * np.exp(-k2_low * time_range)
        )
        
        C_EPA_high = np.where(
            time_range <= 5,
            Cl0 * np.exp(-k1_high * time_range),
            Cl0 * np.exp(5 * (k2_high - k1_high)) * np.exp(-k2_high * time_range)
        )
        
        # 그래프 그리기
        plt.figure(figsize=(5, 3))
        plt.plot(time_range, C_EPA_varied, label='실측데이터 (Virtually Generated)', color='blue', linewidth=3.5)
        # plt.plot(time_range, C_Two_phase, label='Two-phase Model (Original Input)', color='green', linewidth=2.5)
        plt.plot(time_range, C_EPA_low, label='EPA Model Low (User Input)', color='orange', linestyle='--', linewidth=2.5)
        plt.plot(time_range, C_EPA_high, label='EPA Model High (User Input)', color='red', linestyle='--', linewidth=2.5)
        plt.xlabel('Time (hrs)')
        plt.ylabel('Residual Chlorine (mg/L)')
        plt.title('EPA and Two-phase Models of Residual Chlorine')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
        
        # 결과 체크
        is_normal = np.all((C_EPA_varied >= C_EPA_low) & (C_EPA_varied <= C_EPA_high))
        is_initial_phase = time_range <= 0.5
        if is_normal or np.all(is_initial_phase):
            st.subheader("결과: 정상")
            st.markdown("<h1 style='text-align: center; color: green;'>정상</h1>", unsafe_allow_html=True)
        else:
            st.subheader("결과: 비정상")
            st.markdown("<h1 style='text-align: center; color: red;'>비정상</h1>", unsafe_allow_html=True)

else:
    # Disinfection 외의 프로세스
    selected_process_name = st.session_state.selected_process.split(" ", 1)[1]
    timeseries_df = get_timeseries_data(selected_process_name)
    
    st.subheader(f"📌 {st.session_state.selected_process} Details and Data")
    st.markdown(process_descriptions.get(st.session_state.selected_process, "Select a process from the sidebar."))
    st.plotly_chart(create_timeseries_chart(timeseries_df, selected_process_name), use_container_width=True)

# 푸터
st.markdown("---")
st.markdown("ⓒ 2025 K-water AI Lab | Contact: sunghoonkim@kwater.or.kr")
