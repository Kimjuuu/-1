import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 예측기", layout="wide")

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 비교 및 예측기")
st.write(
    "1908년부터의 서울 기온 데이터를 활용하여 전체 데이터 및 학습 기간별(최근 50년 vs 100년) "
    "선형회귀 모델의 예측 성능을 비교합니다."
)


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 연평균기온 계산
    yearly = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly[
        (yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)
    ].copy()
    filtered["X"] = filtered["연도"] - 1908  # 1908년부터 경과 연수

    return filtered


try:
    df = load_and_preprocess()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 데이터셋 분할 (학습 및 테스트)
train_50 = df[(df["연도"] >= 1956) & (df["연도"] <= 2005)].copy()
train_100 = df[(df["연도"] >= 1906) & (df["연도"] <= 2005)].copy()
test_20 = df[(df["연도"] >= 2006) & (df["연도"] <= 2025)].copy()


# 3. 선형회귀 모델 학습 함수
def fit_model(data):
    X_val = data["X"].values
    Y_val = data["평균기온"].values
    slope, intercept = np.polyfit(X_val, Y_val, 1)
    return slope, intercept


# 회귀 계수 구하기
slope_all, intercept_all = fit_model(df)
slope_50, intercept_50 = fit_model(train_50)
slope_100, intercept_100 = fit_model(train_100)


# 4. 모델 평가 함수 (테스트 데이터 기준)
def evaluate_model(slope, intercept, test_data):
    y_true = test_data["평균기온"].values
    y_pred = slope * test_data["X"].values + intercept

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2, y_pred


# 전체 데이터에 대한 평가
pred_all_self = slope_all * df["X"].values + intercept_all
mae_all = mean_absolute_error(df["평균기온"], pred_all_self)
mse_all = mean_squared_error(df["평균기온"], pred_all_self)
r2_all = r2_score(df["평균기온"], pred_all_self)

# 공통 테스트 데이터(2006~2025) 평가
mae_50, mse_50, r2_50, pred_test_50 = evaluate_model(
    slope_50, intercept_50, test_20
)
mae_100, mse_100, r2_100, pred_test_100 = evaluate_model(
    slope_100, intercept_100, test_20
)

# 5. 요약 메트릭 및 평가 결과 표시
st.subheader("📌 데이터 분할 및 모델 성능 평가 결과")

col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric(
    "전체 데이터",
    f"{len(df)}개 해",
    f"{df['연도'].min()}~{df['연도'].max()}년",
)
col_info2.metric(
    "최근 50년 학습 데이터",
    f"{len(train_50)}개 해",
    f"{train_50['연도'].min()}~{train_50['연도'].max()}년",
)
col_info3.metric(
    "공통 테스트 데이터",
    f"{len(test_20)}개 해",
    f"{test_20['연도'].min()}~{test_20['연도'].max()}년",
)

st.write("")

# 평가 결과 비교 테이블
metrics_df = pd.DataFrame(
    {
        "모델 구분": [
            "전체 데이터 모델 (자체 평가)",
            "최근 50년 학습 모델 (1956~2005)",
            "최근 100년 학습 모델 (1906~2005)",
        ],
        "학습 데이터 기간": ["1908 ~ 2025", "1956 ~ 2005", "1906 ~ 2005"],
        "평가 데이터": ["전체 데이터", "최근 20년 (2006~2025)", "최근 20년 (2006~2025)"],
        "기울기 (°C/년)": [
            f"{slope_all:.5f}",
            f"{slope_50:.5f}",
            f"{slope_100:.5f}",
        ],
        "10년당 상승폭": [
            f"+{slope_all*10:.3f} °C",
            f"+{slope_50*10:.3f} °C",
            f"+{slope_100*10:.3f} °C",
        ],
        "MAE": [f"{mae_all:.4f}", f"{mae_50:.4f}", f"{mae_100:.4f}"],
        "MSE": [f"{mse_all:.4f}", f"{mse_50:.4f}", f"{mse_100:.4f}"],
        "R²": [f"{r2_all:.4f}", f"{r2_50:.4f}", f"{r2_100:.4f}"],
    }
)

st.dataframe(metrics_df, use_container_width=True)

st.markdown("---")

# 6. 연도 선택 슬라이더 및 예측값 출력
st.subheader("🔮 연도별 예상 기온 비교")
selected_year = st.slider(
    "예상 기온을 보고 싶은 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - 1908
pred_val_all = slope_all * selected_x + intercept_all
pred_val_50 = slope_50 * selected_x + intercept_50
pred_val_100 = slope_100 * selected_x + intercept_100

p_col1, p_col2, p_col3 = st.columns(3)
p_col1.metric(
    "전체 모델 예상 기온",
    f"{pred_val_all:.2f} °C",
    f"기울기: +{slope_all*10:.2f}°C/10년",
)
p_col2.metric(
    "최근 50년 모델 예상 기온",
    f"{pred_val_50:.2f} °C",
    f"기울기: +{slope_50*10:.2f}°C/10년",
)
p_col3.metric(
    "최근 100년 모델 예상 기온",
    f"{pred_val_100:.2f} °C",
    f"기울기: +{slope_100*10:.2f}°C/10년",
)

# 7. Plotly 시각화
trend_years = np.arange(1900, 2101)
trend_x = trend_years - 1908

line_all = slope_all * trend_x + intercept_all
line_50 = slope_50 * trend_x + intercept_50
line_100 = slope_100 * trend_x + intercept_100

fig = go.Figure()

# 실제 데이터 산점도 (구분 표시)
fig.add_trace(
    go.Scatter(
        x=df[df["연도"] < 2006]["연도"],
        y=df[df["연도"] < 2006]["평균기온"],
        mode="markers",
        name="과거 관측 데이터 (~2005)",
        marker=dict(color="#1f77b4", size=6, opacity=0.6),
    )
)

fig.add_trace(
    go.Scatter(
        x=test_20["연도"],
        y=test_20["평균기온"],
        mode="markers",
        name="최근 20년 테스트 데이터 (2006~2025)",
        marker=dict(color="#2ca02c", size=8, symbol="circle"),
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=trend_years,
        y=line_all,
        mode="lines",
        name="전체 모델 회귀선",
        line=dict(color="gray", width=2, dash="dash"),
    )
)

fig.add_trace(
    go.Scatter(
        x=trend_years,
        y=line_50,
        mode="lines",
        name="최근 50년 학습 회귀선 (1956~2005)",
        line=dict(color="#d62728", width=2.5),
    )
)

fig.add_trace(
    go.Scatter(
        x=trend_years,
        y=line_100,
        mode="lines",
        name="최근 100년 학습 회귀선 (1906~2005)",
        line=dict(color="#ff7f0e", width=2.5),
    )
)

# 선택된 연도 포인트
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[pred_val_50],
        mode="markers",
        name=f"선택 연도({selected_year}년, 50년 모델)",
        marker=dict(color="red", size=12, symbol="star"),
    )
)

fig.update_layout(
    title="서울 연평균 기온 추세 및 모델별 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(range=[1895, 2105]),
    hovermode="x unified",
    template="plotly_white",
    height=550,
)

st.plotly_chart(fig, use_container_width=True)
