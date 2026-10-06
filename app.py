import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 곡선 회귀 예측기", layout="wide")

st.title("📈 서울 연평균 기온 곡선(다항 회귀) 모델 비교 및 예측기")
st.write(
    "2005년 이전 데이터를 훈련용으로, 2005년 이후 데이터를 테스트용으로 분할하여 "
    "1차(직선), 3차 곡선, 9차 곡선 회귀 모델의 예측 성능(MAE)과 2050년 예상 기온을 비교합니다."
)


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 조건: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly[
        (yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)
    ].copy()

    # 고차 곡선 연산의 안정성을 위해 연도를 스케일링 (1908년 = 0)
    base_year = 1908
    filtered["X"] = filtered["연도"] - base_year

    return filtered, base_year


try:
    df, BASE_YEAR = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 훈련용(2005년 이전) / 테스트용(2005년 이후) 분할
train_df = df[df["연도"] < 2005].copy()
test_df = df[df["연도"] >= 2005].copy()

n_train = len(train_df)
n_test = len(test_df)

# 화면 상단에 훈련/테스트 데이터 개수 표기
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("훈련용 데이터 연도 수 (< 2005년)", f"{n_train}개 해")
col_m2.metric(
    "훈련 기간", f"{train_df['연도'].min()} ~ {train_df['연도'].max()}년"
)
col_m3.metric("테스트용 데이터 연도 수 (≥ 2005년)", f"{n_test}개 해")
col_m4.metric("테스트 기간", f"{test_df['연도'].min()} ~ {test_df['연도'].max()}년")

st.markdown("---")

# 3. 다항 회귀 모델 학습 및 평가 (1차, 3차, 9차)
degrees = [1, 3, 9]
results = []
fitted_models = {}

X_train = train_df["X"].values
y_train = train_df["평균기온"].values

X_test = test_df["X"].values
y_test = test_df["평균기온"].values

# 2050년 스케일링된 X 값
x_2050 = 2050 - BASE_YEAR

for deg in degrees:
    # 훈련용 데이터로만 다항식 계수 추정
    coeffs = np.polyfit(X_train, y_train, deg)
    poly_func = np.poly1d(coeffs)
    fitted_models[deg] = poly_func

    # 반드시 학습에 사용하지 않은 테스트 데이터로 예측 및 채점
    y_test_pred = poly_func(X_test)

    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    # 2050년 예상 기온
    pred_2050 = poly_func(x_2050)

    results.append(
        {
            "차수": f"{deg}차 ({'직선' if deg == 1 else '곡선'})",
            "테스트 MAE (평균 빗나간 온도)": f"{mae:.3f} °C",
            "테스트 RMSE": f"{rmse:.3f} °C",
            "2050년 예상 기온": f"{pred_2050:.2f} °C",
        }
    )

# 4. 결과 비교 표 출력
st.subheader("📊 차수별 테스트 데이터 평가 결과 및 2050년 예측값 비교")
results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

st.info(
    "💡 **참고**: 9차와 같은 고차 다항식은 훈련용 데이터 구간 안에서는 데이터를 지나치게 맞춰 오차(오버피팅)가 발생하며, "
    "학습 구간을 벗어난 외삽(2005년 이후 테스트 데이터 및 2050년 예측) 시 기하급수적으로 폭증하거나 급락하는 특성을 보입니다."
)

st.markdown("---")

# 5. 연도 선택 슬라이더 및 예측값 확인
st.subheader("🔮 슬라이더 연도별 모델별 예상 기온")
selected_year = st.slider(
    "예상 기온을 확인하고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
)

sel_x = selected_year - BASE_YEAR
cols = st.columns(len(degrees))

for idx, deg in enumerate(degrees):
    pred_val = fitted_models[deg](sel_x)
    cols[idx].metric(
        label=f"{deg}차 모델 ({selected_year}년)",
        value=f"{pred_val:.2f} °C",
    )

# 6. Plotly 시각화
trend_years = np.arange(1900, 2101)
trend_x = trend_years - BASE_YEAR

fig = go.Figure()

# 훈련용 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_df["평균기온"],
        mode="markers",
        name="훈련용 데이터 (<2005년)",
        marker=dict(color="#1f77b4", size=6, opacity=0.7),
    )
)

# 테스트용 데이터 산점도
fig.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["평균기온"],
        mode="markers",
        name="테스트용 데이터 (≥2005년)",
        marker=dict(color="#2ca02c", size=8, symbol="diamond"),
    )
)

# 모델별 곡선 시각화
colors = {1: "#ff7f0e", 3: "#d62728", 9: "#9467bd"}
dash_styles = {1: "dash", 3: "solid", 9: "dot"}

for deg in degrees:
    y_curve = fitted_models[deg](trend_x)
    fig.add_trace(
        go.Scatter(
            x=trend_years,
            y=y_curve,
            mode="lines",
            name=f"{deg}차 {'직선' if deg == 1 else '곡선'}",
            line=dict(color=colors[deg], width=2.5, dash=dash_styles[deg]),
        )
    )

# Y축 범위 조정 (9차 곡선 폭주 대비 시각적 가독성 유지)
fig.update_layout(
    title="서울 연평균 기온 다항 회귀 모델 비교 (1차 vs 3차 vs 9차)",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(range=[1895, 2105]),
    yaxis=dict(range=[5, 25]),  # 9차 곡선의 과도한 발산 시 visual clipping
    hovermode="x unified",
    template="plotly_white",
    height=600,
)

st.plotly_chart(fig, use_container_width=True)
