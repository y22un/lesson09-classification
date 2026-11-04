import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from itertools import permutations
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)


# --------------------------------------------------
# 1. 페이지 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기",
    page_icon="⚖️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "y22un/data/refs/heads/main/stroke.csv"
)

FEATURE_LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당 수치",
    "hypertension": "고혈압 여부",
    "heart_disease": "심장병 여부",
    "bmi": "체질량지수",
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했어. 인터넷 연결과 데이터 주소를 확인해 줘.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# 2. 화면 제목 및 입력 속성 선택
# --------------------------------------------------

st.title("⚖️ 성능 높이기")

st.markdown(
    """
    뇌졸중을 겪은 사람은 전체 데이터에서 상대적으로 적어.
    그래서 모델이 대부분의 사람을 뇌졸중이 없는 사람이라고 예측해도
    정확도가 높게 나올 수 있어.

    이번에는 **뇌졸중 여부에 따른 가중치를 조정하기 전과 후**를 비교해 보자.
    """
)

st.subheader("1. 입력으로 사용할 속성 선택")

selected_features = st.multiselect(
    "모델이 판단할 때 사용할 속성을 골라 줘.",
    options=list(FEATURE_LABELS.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_LABELS[x],
    key="performance_features",
)

if len(selected_features) < 2:
    st.warning("산점도를 그리려면 속성을 2개 이상 선택해 줘.")
    st.stop()

st.caption(
    "데이터 분리 방식은 '채점표' 페이지와 같아. "
    "id 순으로 정렬한 뒤, 10명마다 앞의 3명을 시험용으로 사용하고 "
    "나머지 7명을 훈련용으로 사용해."
)

st.caption(
    "체질량지수에 빈칸이 있다면 훈련용 데이터에서 계산한 중앙값으로 채워. "
    "시험용 데이터의 정보가 중앙값 계산에 섞이지 않도록 했어."
)


# --------------------------------------------------
# 3. 데이터 전처리 및 훈련/시험 데이터 분리
# --------------------------------------------------

required_columns = ["id", "stroke"] + selected_features

work_df = df[required_columns].copy()

work_df["id"] = pd.to_numeric(work_df["id"], errors="coerce")
work_df["stroke"] = pd.to_numeric(work_df["stroke"], errors="coerce")

work_df = work_df.dropna(subset=["id", "stroke"])
work_df = work_df[work_df["stroke"].isin([0, 1])].copy()

for feature in selected_features:
    work_df[feature] = pd.to_numeric(
        work_df[feature], errors="coerce"
    )

work_df = (
    work_df.sort_values("id", kind="stable")
    .reset_index(drop=True)
)

if len(work_df) < 10:
    st.error("데이터가 너무 적어서 훈련용과 시험용으로 나누기 어려워.")
    st.stop()

# 채점표 페이지와 동일하게 10명마다 앞의 3명을 시험용으로 사용
test_mask = np.array(
    [position % 10 < 3 for position in range(len(work_df))]
)

train_df = work_df.loc[~test_mask].copy()
test_df = work_df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if y_train.nunique() < 2:
    st.error("훈련용 데이터에 뇌졸중 여부가 한 종류만 있어 모델을 학습할 수 없어.")
    st.stop()

if y_test.nunique() < 2:
    st.warning(
        "시험용 데이터에 뇌졸중 여부가 한 종류만 있어. "
        "일부 성능 지표를 해석할 때 주의해 줘."
    )

if y_test.sum() == 0:
    st.warning(
        "시험용 데이터에 실제 뇌졸중 환자가 없어. "
        "환자를 얼마나 찾아냈는지 평가하기 어려워."
    )

st.subheader("2. 데이터 분리 결과")

col1, col2, col3 = st.columns(3)

col1.metric("전체 데이터", f"{len(work_df):,}명")
col2.metric("훈련용 데이터", f"{len(train_df):,}명")
col3.metric("시험용 데이터", f"{len(test_df):,}명")

st.caption(
    f"훈련용 실제 뇌졸중 환자: {int(y_train.sum()):,}명 · "
    f"시험용 실제 뇌졸중 환자: {int(y_test.sum()):,}명"
)


# --------------------------------------------------
# 4. 결측치 처리
# --------------------------------------------------

# 시각화에도 훈련용 데이터로 계산한 중앙값을 사용
plot_imputer = SimpleImputer(strategy="median")

X_train_array = plot_imputer.fit_transform(X_train)
X_test_array = plot_imputer.transform(X_test)

X_train_plot = pd.DataFrame(
    X_train_array,
    columns=selected_features,
    index=X_train.index,
)

X_test_plot = pd.DataFrame(
    X_test_array,
    columns=selected_features,
    index=X_test.index,
)

# 모델 입력에 사용할 데이터도 동일하게 구성
X_train_model = X_train.copy()
X_test_model = X_test.copy()


# --------------------------------------------------
# 5. 모델 생성 및 학습
# --------------------------------------------------

def make_pipeline(model_type, use_class_weight):
    """
    use_class_weight=False: 가중치 조정 전
    use_class_weight=True: 클래스 가중치 조정 후
    """

    class_weight = "balanced" if use_class_weight else None

    steps = []

    if "bmi" in selected_features:
        steps.append(
            ("결측치 처리", SimpleImputer(strategy="median"))
        )

    if model_type == "logistic":
        steps.append(("표준화", StandardScaler()))
        steps.append(
            (
                "모델",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=42,
                ),
            )
        )

    elif model_type == "tree":
        steps.append(
            (
                "모델",
                DecisionTreeClassifier(
                    max_depth=3,
                    min_samples_leaf=5,
                    class_weight=class_weight,
                    random_state=42,
                ),
            )
        )

    return Pipeline(steps)


model_configs = [
    {
        "key": "logistic",
        "name": "로지스틱 회귀(확률로 답하는 모델)",
    },
    {
        "key": "tree",
        "name": "의사결정트리(질문으로 답하는 모델)",
    },
]

models = {}
results = {}

for config in model_configs:
    model_key = config["key"]

    models[model_key] = {}

    for weight_key, use_weight in [
        ("before", False),
        ("after", True),
    ]:
        model = make_pipeline(model_key, use_weight)
        model.fit(X_train_model, y_train)

        predictions = model.predict(X_test_model)
        probabilities = model.predict_proba(X_test_model)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        results[(model_key, weight_key)] = {
            "accuracy": accuracy_score(y_test, predictions),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "predictions": predictions,
            "probabilities": probabilities,
        }

        models[model_key][weight_key] = model


# --------------------------------------------------
# 6. 가중치 조정 전후 성능 비교표
# --------------------------------------------------

st.divider()
st.subheader("3. 가중치 조정 전후 성능 비교")

st.markdown(
    """
    **가중치 조정 전**에는 두 종류의 데이터를 같은 방식으로 취급해.
    **가중치 조정 후**에는 훈련용 데이터의 클래스 분포를 고려해
    상대적으로 드문 뇌졸중 환자의 오분류에 더 큰 가중치를 줘.

    아래 표는 두 모델이 동일한 시험용 데이터를 예측한 결과야.
    """
)


def make_comparison_table(model_key):
    before = results[(model_key, "before")]
    after = results[(model_key, "after")]

    rows = [
        {
            "항목": "TN · 실제 0, 예측 0",
            "가중치 조정 전": before["tn"],
            "가중치 조정 후": after["tn"],
        },
        {
            "항목": "FP · 실제 0, 예측 1",
            "가중치 조정 전": before["fp"],
            "가중치 조정 후": after["fp"],
        },
        {
            "항목": "FN · 실제 1, 예측 0",
            "가중치 조정 전": before["fn"],
            "가중치 조정 후": after["fn"],
        },
        {
            "항목": "TP · 실제 1, 예측 1",
            "가중치 조정 전": before["tp"],
            "가중치 조정 후": after["tp"],
        },
        {
            "항목": "정확도",
            "가중치 조정 전": f'{before["accuracy"]:.1%}',
            "가중치 조정 후": f'{after["accuracy"]:.1%}',
        },
        {
            "항목": "재현율",
            "가중치 조정 전": f'{before["recall"]:.1%}',
            "가중치 조정 후": f'{after["recall"]:.1%}',
        },
        {
            "항목": "정밀도",
            "가중치 조정 전": f'{before["precision"]:.1%}',
            "가중치 조정 후": f'{after["precision"]:.1%}',
        },
        {
            "항목": "F1 점수",
            "가중치 조정 전": f'{before["f1"]:.1%}',
            "가중치 조정 후": f'{after["f1"]:.1%}',
        },
    ]

    return pd.DataFrame(rows)


for config in model_configs:
    model_key = config["key"]

    st.markdown(f"#### {config['name']}")

    comparison = make_comparison_table(model_key)

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )

    before = results[(model_key, "before")]
    after = results[(model_key, "after")]

    st.markdown(
        "**환자 수 변화:** "
        f"찾아낸 환자(TP) {before['tp']}명 → {after['tp']}명 · "
        f"놓친 환자(FN) {before['fn']}명 → {after['fn']}명 · "
        f"헛짚은 사람(FP) {before['fp']}명 → {after['fp']}명"
    )


with st.expander("표에 사용된 네 칸과 네 지표 알아보기"):
    st.markdown(
        """
        **혼동 행렬의 네 칸**

        - **TN:** 뇌졸중이 없는 사람을 없다고 올바르게 예측
        - **FP:** 뇌졸중이 없는 사람을 있다고 잘못 예측
        - **FN:** 실제 환자를 없다고 잘못 예측
        - **TP:** 실제 환자를 있다고 올바르게 예측

        **네 가지 성능 지표**

        - **정확도:** 전체 예측 중 맞힌 비율
        - **재현율:** 실제 환자 중 모델이 찾아낸 비율
        - **정밀도:** 환자라고 예측한 사람 중 실제 환자의 비율
        - **F1 점수:** 정밀도와 재현율을 함께 고려한 지표

        가중치 조정 후 재현율이 높아지더라도 FP가 증가하거나
        정확도가 낮아질 수 있어. 여러 지표를 함께 비교하는 게 중요해.
        """
    )


# --------------------------------------------------
# 7. 시각화용 함수
# --------------------------------------------------

CLASS_COLORS = {
    0: "#2878B5",
    1: "#D9534F",
}

BOUNDARY_COLORS = {
    "before": "#E69F00",
    "after": "#16A085",
}


def get_reference_values():
    """훈련용 데이터에서 각 속성의 중앙값을 구한다."""
    return X_train_plot.median().to_dict()


def make_prediction_frame(values, reference):
    """선택한 축 이외의 속성은 훈련용 중앙값으로 고정한다."""

    # 입력된 배열의 실제 행 개수를 구한다.
    first_value = np.asarray(next(iter(values.values())))
    n_rows = first_value.size if first_value.ndim > 0 else 1

    # 모든 속성을 같은 행 개수로 초기화한다.
    frame = pd.DataFrame(
        {
            feature: np.full(n_rows, reference[feature])
            for feature in selected_features
        }
    )

    # 시각화할 축의 값으로 해당 열을 덮어쓴다.
    for feature, value in values.items():
        value_array = np.asarray(value).reshape(-1)

        if len(value_array) != n_rows:
            raise ValueError(
                f"{FEATURE_LABELS.get(feature, feature)}의 "
                "예측용 데이터 길이가 일치하지 않아."
            )

        frame[feature] = value_array

    return frame



def add_test_points_2d(fig, x_feature, y_feature):
    for class_value in [0, 1]:
        mask = y_test.to_numpy() == class_value

        fig.add_trace(
            go.Scatter(
                x=X_test_plot.loc[mask, x_feature],
                y=X_test_plot.loc[mask, y_feature],
                mode="markers",
                name=(
                    "실제 뇌졸중 없음"
                    if class_value == 0
                    else "실제 뇌졸중 있음"
                ),
                marker=dict(
                    color=CLASS_COLORS[class_value],
                    size=7,
                    opacity=0.7,
                ),
                legendgroup=f"class_{class_value}",
            )
        )


def add_boundary_2d(
    fig,
    model,
    weight_key,
    x_feature,
    y_feature,
    reference,
):
    x_min = float(X_test_plot[x_feature].min())
    x_max = float(X_test_plot[x_feature].max())
    y_min = float(X_test_plot[y_feature].min())
    y_max = float(X_test_plot[y_feature].max())

    # 모든 시험용 데이터가 같은 값인 경우에도 그래프 범위를 확보
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    x_values = np.linspace(x_min, x_max, 100)
    y_values = np.linspace(y_min, y_max, 100)

    xx, yy = np.meshgrid(x_values, y_values)

    frame = make_prediction_frame(
        {
            x_feature: xx.ravel(),
            y_feature: yy.ravel(),
        },
        reference,
    )

    probabilities = model.predict_proba(frame)[:, 1]
    probability_grid = probabilities.reshape(xx.shape)

    min_probability = float(np.min(probability_grid))
    max_probability = float(np.max(probability_grid))

    boundary_exists = (
        min_probability <= 0.5 <= max_probability
    )

    color = BOUNDARY_COLORS[weight_key]
    label = (
        "가중치 조정 전 경계선"
        if weight_key == "before"
        else "가중치 조정 후 경계선"
    )

    if boundary_exists:
        fig.add_trace(
            go.Contour(
                x=x_values,
                y=y_values,
                z=probability_grid,
                name=label,
                showscale=False,
                hoverinfo="skip",
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=0.01,
                    coloring="none",
                    showlabels=False,
                ),
                line=dict(
                    color=color,
                    width=3,
                    dash="dash" if weight_key == "before" else "solid",
                ),
            )
        )

    return {
        "exists": boundary_exists,
        "minimum": min_probability,
        "maximum": max_probability,
        "label": label,
    }


def add_boundary_3d(
    fig,
    model,
    weight_key,
    x_feature,
    y_feature,
    z_feature,
    reference,
):
    axis_features = [x_feature, y_feature, z_feature]

    axis_values = {}

    for feature in axis_features:
        low = float(X_test_plot[feature].min())
        high = float(X_test_plot[feature].max())

        if low == high:
            low -= 0.5
            high += 0.5

        axis_values[feature] = np.linspace(low, high, 18)

    xx, yy, zz = np.meshgrid(
        axis_values[x_feature],
        axis_values[y_feature],
        axis_values[z_feature],
        indexing="ij",
    )

    frame = make_prediction_frame(
        {
            x_feature: xx.ravel(),
            y_feature: yy.ravel(),
            z_feature: zz.ravel(),
        },
        reference,
    )

    probabilities = model.predict_proba(frame)[:, 1]

    minimum = float(np.min(probabilities))
    maximum = float(np.max(probabilities))

    # 0.5 부근의 등가면을 경계면으로 표시
    boundary_exists = minimum <= 0.5 <= maximum

    if boundary_exists:
        color = BOUNDARY_COLORS[weight_key]

        label = (
            "가중치 조정 전 0.5 경계면"
            if weight_key == "before"
            else "가중치 조정 후 0.5 경계면"
        )

        fig.add_trace(
            go.Isosurface(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=probabilities,
                isomin=0.49,
                isomax=0.51,
                surface_count=1,
                opacity=0.55,
                colorscale=[
                    [0, color],
                    [1, color],
                ],
                showscale=False,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False,
                ),
                name=label,
                hovertemplate=(
                    f"{FEATURE_LABELS[x_feature]}: %{{x:.2f}}"
                    f"<br>{FEATURE_LABELS[y_feature]}: %{{y:.2f}}"
                    f"<br>{FEATURE_LABELS[z_feature]}: %{{z:.2f}}"
                    "<extra>0.5 경계면</extra>"
                ),
            )
        )

    return {
        "exists": boundary_exists,
        "minimum": minimum,
        "maximum": maximum,
    }


def add_test_points_3d(fig, x_feature, y_feature, z_feature):
    for class_value in [0, 1]:
        mask = y_test.to_numpy() == class_value

        fig.add_trace(
            go.Scatter3d(
                x=X_test_plot.loc[mask, x_feature],
                y=X_test_plot.loc[mask, y_feature],
                z=X_test_plot.loc[mask, z_feature],
                mode="markers",
                name=(
                    "실제 뇌졸중 없음"
                    if class_value == 0
                    else "실제 뇌졸중 있음"
                ),
                marker=dict(
                    color=CLASS_COLORS[class_value],
                    size=4,
                    opacity=0.7,
                ),
            )
        )


# --------------------------------------------------
# 8. 축 선택
# --------------------------------------------------

st.divider()
st.subheader("4. 가중치 조정 전후 경계 비교")

st.markdown(
    """
    점의 색은 실제 뇌졸중 여부를 나타내고, 경계선 또는 경계면은
    모델이 뇌졸중 확률을 0.5로 판단하는 지점을 나타내.

    주황색 점선은 가중치 조정 전, 초록색 실선 또는 면은 조정 후야.
    """
)

reference_values = get_reference_values()

if len(selected_features) == 2:
    axis_combinations = list(permutations(selected_features, 2))

    selected_axes = st.selectbox(
        "가로축 / 세로축 선택",
        options=axis_combinations,
        format_func=lambda axes: (
            f"{FEATURE_LABELS[axes[0]]} / "
            f"{FEATURE_LABELS[axes[1]]}"
        ),
        key="performance_axes_2d",
    )

    x_feature, y_feature = selected_axes

    for config in model_configs:
        model_key = config["key"]

        st.markdown(f"#### {config['name']}")

        fig = go.Figure()

        add_test_points_2d(
            fig,
            x_feature,
            y_feature,
        )

        before_boundary = add_boundary_2d(
            fig,
            models[model_key]["before"],
            "before",
            x_feature,
            y_feature,
            reference_values,
        )

        after_boundary = add_boundary_2d(
            fig,
            models[model_key]["after"],
            "after",
            x_feature,
            y_feature,
            reference_values,
        )

        fig.update_layout(
            xaxis_title=FEATURE_LABELS[x_feature],
            yaxis_title=FEATURE_LABELS[y_feature],
            template="plotly_white",
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=20, r=20, t=70, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        if not before_boundary["exists"]:
            st.info(
                "가중치 조정 전의 0.5 경계선은 "
                "현재 그림의 범위 안에 나타나지 않아. "
                f"이 범위에서 예측 확률은 "
                f"{before_boundary['minimum']:.3f}~"
                f"{before_boundary['maximum']:.3f}이야."
            )

        if not after_boundary["exists"]:
            st.info(
                "가중치 조정 후의 0.5 경계선은 "
                "현재 그림의 범위 안에 나타나지 않아. "
                f"이 범위에서 예측 확률은 "
                f"{after_boundary['minimum']:.3f}~"
                f"{after_boundary['maximum']:.3f}이야."
            )

else:
    axis_combinations = list(permutations(selected_features, 3))

    selected_axes = st.selectbox(
        "가로축 / 세로축 / 깊이축 선택",
        options=axis_combinations,
        format_func=lambda axes: (
            f"{FEATURE_LABELS[axes[0]]} / "
            f"{FEATURE_LABELS[axes[1]]} / "
            f"{FEATURE_LABELS[axes[2]]}"
        ),
        key="performance_axes_3d",
    )

    x_feature, y_feature, z_feature = selected_axes

    st.caption(
        "3차원 그림은 마우스로 드래그해 회전하고, "
        "스크롤로 확대하거나 축소할 수 있어. "
        "선택하지 않은 속성은 훈련용 데이터의 중앙값으로 고정했어."
    )

    for config in model_configs:
        model_key = config["key"]

        st.markdown(f"#### {config['name']}")

        fig = go.Figure()

        add_test_points_3d(
            fig,
            x_feature,
            y_feature,
            z_feature,
        )

        before_boundary = add_boundary_3d(
            fig,
            models[model_key]["before"],
            "before",
            x_feature,
            y_feature,
            z_feature,
            reference_values,
        )

        after_boundary = add_boundary_3d(
            fig,
            models[model_key]["after"],
            "after",
            x_feature,
            y_feature,
            z_feature,
            reference_values,
        )

        fig.update_layout(
            scene=dict(
                xaxis_title=FEATURE_LABELS[x_feature],
                yaxis_title=FEATURE_LABELS[y_feature],
                zaxis_title=FEATURE_LABELS[z_feature],
            ),
            template="plotly_white",
            height=700,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=10, r=10, t=70, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

        if not before_boundary["exists"]:
            st.info(
                "가중치 조정 전의 0.5 경계면이 "
                "현재 그림의 범위 안에 나타나지 않아. "
                f"이 범위에서 예측 확률은 "
                f"{before_boundary['minimum']:.3f}~"
                f"{before_boundary['maximum']:.3f}이야."
            )

        if not after_boundary["exists"]:
            st.info(
                "가중치 조정 후의 0.5 경계면이 "
                "현재 그림의 범위 안에 나타나지 않아. "
                f"이 범위에서 예측 확률은 "
                f"{after_boundary['minimum']:.3f}~"
                f"{after_boundary['maximum']:.3f}이야."
            )


# --------------------------------------------------
# 9. 마무리 안내
# --------------------------------------------------

st.divider()

st.subheader("5. 결과를 해석할 때 기억할 점")

st.markdown(
    """
    - 재현율이 높아지면 실제 환자를 더 많이 찾아냈을 가능성이 있어.
    - 정밀도가 낮아지면 환자라고 예측했지만 실제로는 환자가 아닌 사람,
      즉 헛짚은 사람이 늘었을 수 있어.
    - 정확도 하나만으로 어느 모델이 더 좋다고 결정하지 말고,
      재현율·정밀도·F1 점수와 혼동 행렬을 함께 비교해 봐.
    - 가중치 조정은 모델의 학습 방식을 바꾸는 것이지,
      실제 환자 수를 바꾸는 것은 아니야.

    **주의:** 이 페이지는 분류 모델의 학습과 평가를 위한 실습이야.
    의료 진단이나 개인의 뇌졸중 위험 판단에 사용하면 안 돼.
    """
)
