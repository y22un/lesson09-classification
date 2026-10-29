import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================================
# 1. 페이지 설정 및 데이터 불러오기
# ==========================================

st.title("🔎 데이터 탐색")
st.caption("뇌졸중 데이터의 분포와 변수 사이의 관계를 살펴보는 페이지입니다.")

DATA_URL = "https://raw.githubusercontent.com/y22un/data/refs/heads/main/stroke.csv"


@st.cache_data
def load_data():
    """지정된 CSV 데이터를 UTF-8 인코딩으로 불러옵니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()


# 필요한 열이 있는지 확인
required_columns = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
    "stroke",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error(f"CSV에서 필요한 열을 찾을 수 없습니다: {', '.join(missing_columns)}")
    st.stop()


# 뇌졸중 여부를 그래프에 표시할 한글 이름으로 변환
stroke_labels = {
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
}

df["stroke_label"] = df["stroke"].map(stroke_labels).fillna("기타")

stroke_order = ["뇌졸중 없음", "뇌졸중 있음"]


# ==========================================
# 2. 나이와 평균 혈당의 분포
# ==========================================

st.header("1. 나이와 평균 혈당의 분포")
st.write(
    "히스토그램은 값이 어느 구간에 얼마나 많이 분포하는지 보여 줍니다."
)

col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig_age_hist.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.08,
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
    )
    fig_glucose_hist.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.08,
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)


# ==========================================
# 3. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ==========================================

st.header("2. 뇌졸중 여부에 따른 비교")
st.write(
    "상자그림으로 두 그룹의 값이 어떻게 분포하는지 비교합니다. "
    "아래 표에서는 각 그룹의 평균값도 확인할 수 있습니다."
)

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        category_orders={"stroke_label": stroke_order},
        title="뇌졸중 여부에 따른 나이",
        labels={
            "stroke_label": "뇌졸중 여부",
            "age": "나이",
        },
        points="outliers",
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col2:
    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        category_orders={"stroke_label": stroke_order},
        title="뇌졸중 여부에 따른 평균 혈당",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당",
        },
        points="outliers",
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 그룹별 평균 계산
mean_table = (
    df.groupby("stroke_label", observed=False)
    .agg(
        인원수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(stroke_order)
    .reset_index()
    .rename(columns={"stroke_label": "뇌졸중 여부"})
)

mean_table["평균 나이"] = mean_table["평균_나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균_혈당"].round(2)

mean_table = mean_table[
    ["뇌졸중 여부", "인원수", "평균 나이", "평균 혈당"]
]

st.subheader("그룹별 평균값")
st.dataframe(mean_table, hide_index=True, use_container_width=True)


# ==========================================
# 4. 고혈압 유무에 따른 뇌졸중 비율
# ==========================================

st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")
st.write(
    "각 그룹에서 뇌졸중이 기록된 사람의 비율을 비교합니다. "
    "비율은 해당 그룹의 전체 인원 중 stroke=1인 사람의 비율입니다."
)


def make_rate_table(column, label_name):
    """0과 1로 기록된 변수별 인원수와 뇌졸중 비율을 계산합니다."""
    temp = df[[column, "stroke"]].copy()
    temp[column] = pd.to_numeric(temp[column], errors="coerce")
    temp = temp.dropna(subset=[column, "stroke"])

    result = (
        temp.groupby(column)
        .agg(
            인원수=("stroke", "size"),
            뇌졸중_인원=("stroke", "sum"),
            뇌졸중_비율=("stroke", "mean"),
        )
        .reindex([0, 1])
        .dropna(subset=["인원수"])
        .reset_index()
    )

    result[label_name] = result[column].map(
        {0: "없음", 1: "있음"}
    )

    result["뇌졸중 비율(%)"] = result["뇌졸중_비율"] * 100

    return result


def make_rate_chart(rate_table, label_name, title):
    """그룹별 뇌졸중 비율을 Plotly 막대그래프로 만듭니다."""
    fig = px.bar(
        rate_table,
        x=label_name,
        y="뇌졸중 비율(%)",
        text=rate_table["뇌졸중 비율(%)"].map(lambda value: f"{value:.2f}%"),
        title=title,
        labels={
            label_name: "구분",
            "뇌졸중 비율(%)": "뇌졸중 비율(%)",
        },
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis_title="뇌졸중 비율(%)",
        xaxis_title="",
    )

    return fig


hypertension_table = make_rate_table("hypertension", "고혈압")
heart_disease_table = make_rate_table("heart_disease", "심장병")

col1, col2 = st.columns(2)

with col1:
    fig_hypertension = make_rate_chart(
        hypertension_table,
        "고혈압",
        "고혈압 유무에 따른 뇌졸중 비율",
    )
    st.plotly_chart(fig_hypertension, use_container_width=True)

    st.caption("고혈압 그룹별 집계")
    st.dataframe(
        hypertension_table[
            ["고혈압", "인원수", "뇌졸중_인원", "뇌졸중 비율(%)"]
        ].rename(
            columns={
                "인원수": "전체 인원",
                "뇌졸중_인원": "뇌졸중 인원",
            }
        ).round(2),
        hide_index=True,
        use_container_width=True,
    )

with col2:
    fig_heart_disease = make_rate_chart(
        heart_disease_table,
        "심장병",
        "심장병 유무에 따른 뇌졸중 비율",
    )
    st.plotly_chart(fig_heart_disease, use_container_width=True)

    st.caption("심장병 그룹별 집계")
    st.dataframe(
        heart_disease_table[
            ["심장병", "인원수", "뇌졸중_인원", "뇌졸중 비율(%)"]
        ].rename(
            columns={
                "인원수": "전체 인원",
                "뇌졸중_인원": "뇌졸중 인원",
            }
        ).round(2),
        hide_index=True,
        use_container_width=True,
    )


# ==========================================
# 5. BMI 결측값이 있는 사람들의 뇌졸중 비율
# ==========================================

st.header("4. BMI 결측값과 뇌졸중 비율")
st.write(
    "체질량지수(bmi)가 비어 있는 사람들만 따로 살펴보고, "
    "전체 데이터의 뇌졸중 비율과 비교합니다."
)

missing_bmi_df = df[df["bmi"].isna()]
total_count = len(df)
missing_bmi_count = len(missing_bmi_df)

total_stroke_rate = (
    df["stroke"].eq(1).mean() * 100
    if total_count > 0
    else float("nan")
)

missing_bmi_stroke_rate = (
    missing_bmi_df["stroke"].eq(1).mean() * 100
    if missing_bmi_count > 0
    else float("nan")
)

bmi_comparison = pd.DataFrame(
    [
        {
            "구분": "BMI가 비어 있는 사람",
            "인원수": missing_bmi_count,
            "뇌졸중 인원": int(missing_bmi_df["stroke"].eq(1).sum()),
            "뇌졸중 비율(%)": missing_bmi_stroke_rate,
        },
        {
            "구분": "전체 사람",
            "인원수": total_count,
            "뇌졸중 인원": int(df["stroke"].eq(1).sum()),
            "뇌졸중 비율(%)": total_stroke_rate,
        },
    ]
)

st.dataframe(
    bmi_comparison.round(2),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "BMI 결측 여부와 뇌졸중 비율의 차이는 관찰된 관계입니다. "
    "이 비교만으로 원인이나 인과관계를 판단할 수는 없습니다."
)


# ==========================================
# 6. 흡연 상태별 인원수
# ==========================================

st.header("5. 흡연 상태별 인원수")

smoking_counts = (
    df["smoking_status"]
    .fillna("결측값")
    .astype(str)
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="인원수")
)

st.dataframe(
    smoking_counts,
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "이 페이지의 결과는 제공된 데이터에 대한 탐색적 분석이며, "
    "개인의 건강 상태를 진단하거나 뇌졸중 위험을 예측하는 결과가 아닙니다."
)
