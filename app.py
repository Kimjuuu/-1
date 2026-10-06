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
test_df = df[df["연도"] >=
