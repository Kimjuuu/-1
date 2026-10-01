import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="서울 기온 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 예측기")
st.write("서울의 과거 기온 데이터를 바탕으로 연도별 평균기온을 추정하고 미래 기온을 예측합니다.")


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    # 데이터 읽기 (인코딩 UTF-8)
    df = pd.read_csv(url, encoding="utf-8")

    # '날짜' 컬럼을 datetime 형식으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 필터링 조건: 2025년 이하 & 관측일수 300일 이상
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].copy()

    return filtered_df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 회귀분석 계산 (1908년부터 지난 연수를 독립변수 X로 지정)
# X = 연도 - 1908
df["X"] = df["연도"] - 1908
X = df["X"].values
Y = df["평균기온"].values

# 선형 회귀 계수 구하기 (Y = slope * X + intercept)
slope, intercept = np.polyfit(X, Y, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(X, Y)
correlation = corr_matrix[0, 1]

# 3. 데이터 정보 및 메트릭 출력
start_year = int(df["연도"].min())
end_year = int(df["연도"].max())
total_years = len(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("학습 데이터 연도 수", f"{total_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수", f"{correlation:.4f}")

st.markdown("---")

# 4. 연도 선택 및 예상 기온 예측
selected_year = st.slider(
    "예상 기온을 보고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

# 선택한 연도의 독립 변수 계산 및 기온 예측
selected_x = selected_year - 1908
predicted_temp = slope * selected_x + intercept

st.subheader(f"🔮 {selected_year}년 서울 예상 연평균 기온")
st.metric(label="", value=f"{predicted_temp:.2f} °C")

# 5. Plotly 그래프 작성
# 전체 시각화 범위(1900~2100)에 따른 회귀선 생성
trend_years = np.arange(1900, 2101)
trend_x = trend_years - 1908
trend_y = slope * trend_x + intercept

fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["평균기온"],
        mode="markers",
        name="실제 연평균 기온",
        marker=dict(color="#1f77b4", size=7, opacity=0.8),
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=trend_years,
        y=trend_y,
        mode="lines",
        name="선형 회귀선",
        line=dict(color="#ff7f0e", width=2),
    )
)

# 선택한 연도 예측값 강조 점
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"선택 연도({selected_year}년) 예측값",
        marker=dict(color="red", size=12, symbol="star"),
    )
)

fig.update_layout(
    title="서울 연평균 기온 변화 및 추세선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(range=[1895, 2105]),
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig, use_container_width=True)
