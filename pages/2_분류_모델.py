import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from itertools import product
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


# ==========================================
# 1. 페이지 설정
# ==========================================

st.title("🧠 분류 모델")
st.caption(
    "데이터를 이용해 뇌졸중 여부를 분류하고, "
    "두 가지 모델의 답변과 정확도를 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "y22un/data/refs/heads/main/stroke.csv"
)

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

FEATURE_COLUMNS = list(FEATURE_NAMES.keys())

STROKE_NAMES = {
    0: "아님",
    1: "뇌졸중",
}

RANDOM_STATE = 42


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    """CSV 데이터를 UTF-8로 불러옵니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. CSV 주소와 인터넷 연결을 확인해 주세요.")
    st.exception(error)
    st.stop()


required_columns = ["id", "stroke"] + FEATURE_COLUMNS
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "CSV에서 필요한 열을 찾을 수 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# 번호와 정답을 확인하고 번호순으로 정렬합니다.
df["id"] = pd.to_numeric(df["id"], errors="coerce")
df["stroke"] = pd.to_numeric(df["stroke"], errors="coerce")

df = df.dropna(subset=["id", "stroke"]).copy()
df = df[df["stroke"].isin([0, 1])].copy()
df["stroke"] = df["stroke"].astype(int)

df = df.sort_values("id", kind="stable").reset_index(drop=True)

# 분석에 사용할 속성을 숫자로 변환합니다.
for column in FEATURE_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# ==========================================
# 3. 입력 속성 선택
# ==========================================

st.header("1. 입력 속성 선택")

st.write(
    "모델이 뇌졸중 여부를 판단할 때 사용할 속성을 골라 주세요. "
    "처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다."
)

default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

selected_features = st.multiselect(
    "분류에 사용할 속성",
    options=FEATURE_COLUMNS,
    default=default_features,
    format_func=lambda column: FEATURE_NAMES[column],
    help="두 개 이상의 속성을 선택해야 모델을 학습할 수 있습니다.",
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()


# ==========================================
# 4. 학습 데이터와 테스트 데이터 분리
# ==========================================

st.header("2. 학습 데이터와 테스트 데이터")

st.write(
    "번호순으로 정렬한 뒤 10명씩 묶습니다. "
    "각 묶음의 앞 세 명은 테스트용, 나머지 일곱 명은 학습용으로 사용합니다."
)

# 각 10명 묶음에서 앞의 3명만 테스트용으로 고정합니다.
# 예: 0~2번째는 테스트, 3~9번째는 학습,
#     10~12번째는 테스트, 13~19번째는 학습 ...
test_mask = pd.Series(
    [position % 10 < 3 for position in range(len(df))]
)

X = df[selected_features].copy()
y = df["stroke"].copy()

X_train = X.loc[~test_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[~test_mask].copy()
y_test = y.loc[test_mask].copy()

st.info(
    f"전체 데이터: {len(df):,}명  |  "
    f"학습 데이터: {len(X_train):,}명  |  "
    f"테스트 데이터: {len(X_test):,}명"
)

if len(X_test) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(X_test):,}명입니다. "
        "요청한 1,533명과 다르다면 CSV의 행 수나 전처리 결과를 확인해 주세요."
    )


# ==========================================
# 5. 모델 학습
# ==========================================

st.header("3. 세 가지 모델 학습")

st.write(
    "로지스틱 회귀와 의사결정트리를 학습합니다. "
    "기준 모델은 입력 속성을 사용하지 않고 학습 데이터에서 더 많은 "
    "클래스만 답합니다."
)

# BMI를 선택했을 때만 중앙값 대체를 적용합니다.
# SimpleImputer는 fit 과정에서 학습 데이터의 중앙값만 계산합니다.
# 테스트 데이터의 결측값에는 학습 데이터에서 계산한 값을 사용합니다.

logistic_steps = []

if "bmi" in selected_features:
    logistic_steps.append(
        ("imputer", SimpleImputer(strategy="median"))
    )

# 로지스틱 회귀는 입력 속성의 크기에 영향을 받을 수 있으므로
# StandardScaler를 사용합니다. scaler는 학습 데이터로만 fit합니다.
logistic_steps.extend([
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    ),
])

logistic_model = Pipeline(logistic_steps)


tree_steps = []

if "bmi" in selected_features:
    tree_steps.append(
        ("imputer", SimpleImputer(strategy="median"))
    )

# 질문은 최대 3번, 마지막 마디의 인원은 최소 5명으로 설정합니다.
tree_steps.append(
    (
        "model",
        DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    )
)

tree_model = Pipeline(tree_steps)


# 기준 모델: 학습 데이터에서 더 많은 클래스로만 답합니다.
baseline_model = DummyClassifier(
    strategy="most_frequent",
    random_state=RANDOM_STATE,
)

# 기준 모델에는 입력 정보를 전달하지 않도록 상수 열만 사용합니다.
X_train_baseline = pd.DataFrame(
    {"상수": [0] * len(X_train)},
    index=X_train.index,
)
X_test_baseline = pd.DataFrame(
    {"상수": [0] * len(X_test)},
    index=X_test.index,
)


try:
    logistic_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    baseline_model.fit(X_train_baseline, y_train)
except Exception as error:
    st.error(
        "모델 학습 중 오류가 발생했습니다. "
        "선택한 속성의 값과 결측값을 확인해 주세요."
    )
    st.exception(error)
    st.stop()


# ==========================================
# 6. 정확도 계산
# ==========================================

def evaluate_model(model, train_x, test_x):
    """훈련 데이터와 테스트 데이터의 정확도를 계산합니다."""
    train_prediction = model.predict(train_x)
    test_prediction = model.predict(test_x)

    return {
        "train_accuracy": accuracy_score(
            y_train, train_prediction
        ),
        "test_accuracy": accuracy_score(
            y_test, test_prediction
        ),
        "train_prediction": train_prediction,
        "test_prediction": test_prediction,
    }


logistic_result = evaluate_model(
    logistic_model, X_train, X_test
)

tree_result = evaluate_model(
    tree_model, X_train, X_test
)

baseline_result = evaluate_model(
    baseline_model,
    X_train_baseline,
    X_test_baseline,
)


# ==========================================
# 7. 정확도 카드
# ==========================================

st.header("4. 모델별 정확도 비교")

st.caption(
    "정확도는 모델이 맞힌 사람 수를 전체 사람 수로 나눈 값입니다. "
    "큰 숫자는 테스트 데이터의 정확도입니다."
)


def accuracy_card(title, result):
    """테스트 정확도를 크게, 훈련·테스트 정확도를 작게 표시합니다."""
    with st.container(border=True):
        st.markdown(f"**{title}**")

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{result['test_accuracy']:.1%}",
        )

        small_col1, small_col2 = st.columns(2)

        with small_col1:
            st.caption("훈련 데이터")
            st.markdown(
                f"**{result['train_accuracy']:.1%}**"
            )

        with small_col2:
            st.caption("테스트 데이터")
            st.markdown(
                f"**{result['test_accuracy']:.1%}**"
            )


metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_result,
    )

with metric_col2:
    accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_result,
    )

with metric_col3:
    accuracy_card(
        "기준 모델(입력을 보지 않고 다수 클래스로 답하는 모델)",
        baseline_result,
    )


st.caption(
    "훈련 데이터의 정확도와 테스트 데이터의 정확도 차이를 비교하면 "
    "모델이 학습 데이터에 지나치게 맞춰졌는지 살펴보는 데 도움이 됩니다."
)


# ==========================================
# 8. 두 속성으로 분류 결과 시각화
# ==========================================

st.header("5. 두 속성으로 살펴보는 분류 결과")

st.write(
    "가로축과 세로축으로 사용할 속성을 골라 주세요. "
    "점의 색은 테스트 데이터에 실제로 기록된 뇌졸중 여부를 나타냅니다. "
    "배경은 의사결정트리의 예측 영역이고, 검은 선은 로지스틱 회귀의 "
    "예측 확률이 0.5인 경계입니다."
)

axis_col1, axis_col2 = st.columns(2)

with axis_col1:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda column: FEATURE_NAMES[column],
        key="classification_x_axis",
    )

with axis_col2:
    y_default_index = (
        1 if selected_features[1] != x_feature else 0
    )

    y_feature = st.selectbox(
        "세로축",
        options=selected_features,
        index=y_default_index,
        format_func=lambda column: FEATURE_NAMES[column],
        key="classification_y_axis",
    )

if x_feature == y_feature:
    st.warning("가로축과 세로축은 서로 다른 속성을 선택해 주세요.")
    st.stop()


# 테스트 데이터에서 BMI가 선택된 경우,
# 학습 데이터에서 계산한 중앙값으로 결측값을 대체합니다.
if "bmi" in selected_features:
    fitted_imputer = logistic_model.named_steps["imputer"]

    X_test_filled = pd.DataFrame(
        fitted_imputer.transform(X_test),
        columns=selected_features,
        index=X_test.index,
    )
else:
    X_test_filled = X_test.copy()


# 테스트 데이터에서 두 축의 범위를 구합니다.
x_observed = X_test_filled[x_feature].dropna()
y_observed = X_test_filled[y_feature].dropna()

if x_observed.empty or y_observed.empty:
    st.error("선택한 축에 표시할 수 있는 테스트 데이터가 없습니다.")
    st.stop()


def make_axis_values(series, number_of_points=80):
    """축의 최소·최대값 사이에 격자 좌표를 생성합니다."""
    minimum = float(series.min())
    maximum = float(series.max())

    padding = (maximum - minimum) * 0.08

    if padding == 0:
        padding = 1.0

    minimum -= padding
    maximum += padding

    return [
        minimum + (maximum - minimum) * i / (number_of_points - 1)
        for i in range(number_of_points)
    ]


x_values = make_axis_values(x_observed)
y_values = make_axis_values(y_observed)

# 두 축 이외의 속성은 테스트 데이터의 중앙값으로 고정합니다.
# BMI가 축이 아닌 경우에도 이 기준은 테스트 데이터의 관측된 값으로 계산합니다.
test_medians = X_test.median(numeric_only=True)

grid_points = list(product(y_values, x_values))

grid_data = {}

for feature in selected_features:
    if feature == x_feature:
        grid_data[feature] = [
            x_value for y_value, x_value in grid_points
        ]
    elif feature == y_feature:
        grid_data[feature] = [
            y_value for y_value, x_value in grid_points
        ]
    else:
        grid_data[feature] = [
            float(test_medians[feature])
            if pd.notna(test_medians[feature])
            else 0.0
            for _ in grid_points
        ]

grid_df = pd.DataFrame(grid_data, columns=selected_features)

st.markdown("**두 축 이외의 속성을 고정한 값**")

fixed_features = [
    feature
    for feature in selected_features
    if feature not in [x_feature, y_feature]
]

if fixed_features:
    fixed_descriptions = [
        f"{FEATURE_NAMES[feature]}: "
        f"{test_medians[feature]:.3f}"
        for feature in fixed_features
    ]
    st.write(" · ".join(fixed_descriptions))
else:
    st.write("두 축 이외에 고정할 속성이 없습니다.")


# 각 격자 지점에서 두 모델의 예측을 계산합니다.
try:
    grid_tree_prediction = tree_model.predict(grid_df).tolist()

    grid_probability = logistic_model.predict_proba(
        grid_df
    )[:, list(logistic_model.classes_).index(1)].tolist()

except Exception as error:
    st.error("분류 경계 계산 중 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


grid_size = len(x_values)

tree_z = [
    grid_tree_prediction[
        row_index * grid_size:(row_index + 1) * grid_size
    ]
    for row_index in range(len(y_values))
]

probability_z = [
    grid_probability[
        row_index * grid_size:(row_index + 1) * grid_size
    ]
    for row_index in range(len(y_values))
]


# ==========================================
# 9. Plotly 산점도와 모델의 분류 영역
# ==========================================

fig = go.Figure()

# 의사결정트리의 예측 영역을 옅은 색으로 표시합니다.
fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=tree_z,
        zmin=0,
        zmax=1,
        colorscale=[
            [0.0, "rgba(99, 110, 250, 0.22)"],
            [1.0, "rgba(239, 85, 59, 0.22)"],
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 예측 영역",
    )
)


# 로지스틱 회귀의 확률 0.5 경계가 그림 안에 있는지 확인합니다.
probability_min = min(grid_probability)
probability_max = max(grid_probability)

boundary_inside = (
    probability_min <= 0.5 <= probability_max
)

if boundary_inside:
    fig.add_trace(
        go.Contour(
            x=x_values,
            y=y_values,
            z=probability_z,
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 1,
                "coloring": "lines",
                "showlabels": False,
            },
            line={
                "color": "black",
                "width": 3,
            },
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀 확률 0.5 경계",
        )
    )


# 실제 테스트 데이터를 뇌졸중 여부에 따라 나누어 표시합니다.
actual_stroke = y_test.eq(1)
actual_no_stroke = y_test.eq(0)

fig.add_trace(
    go.Scatter(
        x=X_test_filled.loc[actual_no_stroke, x_feature],
        y=X_test_filled.loc[actual_no_stroke, y_feature],
        mode="markers",
        name="실제 뇌졸중 아님 (0)",
        marker={
            "color": "#3366CC",
            "size": 7,
            "opacity": 0.75,
            "line": {"width": 0.5, "color": "white"},
        },
        hovertemplate=(
            f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
            f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
            "실제 결과: 아님 (0)<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=X_test_filled.loc[actual_stroke, x_feature],
        y=X_test_filled.loc[actual_stroke, y_feature],
        mode="markers",
        name="실제 뇌졸중 (1)",
        marker={
            "color": "#E45756",
            "size": 8,
            "opacity": 0.85,
            "symbol": "circle",
            "line": {"width": 0.5, "color": "white"},
        },
        hovertemplate=(
            f"{FEATURE_NAMES[x_feature]}: %{{x:.2f}}<br>"
            f"{FEATURE_NAMES[y_feature]}: %{{y:.2f}}<br>"
            "실제 결과: 뇌졸중 (1)<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="테스트 데이터와 분류 모델의 경계",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650,
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

if boundary_inside:
    st.success(
        "로지스틱 회귀의 확률 0.5 경계가 현재 그림의 범위 안에 있습니다."
    )
else:
    st.warning(
        "로지스틱 회귀의 확률 0.5 경계가 현재 그림의 범위 안에 없습니다. "
        "따라서 이 그림에서는 경계선을 확인할 수 없습니다."
    )

st.caption(
    "배경색은 의사결정트리의 예측 결과이며, 점의 색은 실제 정답입니다. "
    "배경색과 점의 색이 다를 수 있습니다."
)


# ==========================================
# 10. 의사결정트리의 질문 구조
# ==========================================

st.header("6. 의사결정트리가 던지는 질문")

st.write(
    "의사결정트리는 속성과 기준값을 질문하며 데이터를 나눕니다. "
    "질문의 깊이는 최대 3단계이며, 마지막 마디에 훈련용 사람이 "
    "5명 미만으로 남도록 나누지 않습니다."
)

tree_classifier = tree_model.named_steps["model"]
tree_structure = tree_classifier.tree_
tree_classes = list(tree_classifier.classes_)

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", pad="0.3", nodesep="0.35", ranksep="0.55"];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, margin="0.12,0.08"];',
    'edge [fontname="sans-serif", fontsize=10, arrowsize=0.7];',
]

leaf_count = 0
negative_leaf_count = 0
used_feature_indices = []

for node_index in range(tree_structure.node_count):
    left_child = tree_structure.children_left[node_index]
    right_child = tree_structure.children_right[node_index]

    sample_count = int(tree_structure.n_node_samples[node_index])
    class_counts = tree_structure.value[node_index][0].tolist()

    # 클래스 순서는 tree_classifier.classes_에 맞춰 확인합니다.
    positive_count = 0
    negative_count = 0

    for class_index, class_value in enumerate(tree_classes):
        class_count = int(round(class_counts[class_index]))

        if int(class_value) == 1:
            positive_count = class_count
        else:
            negative_count = class_count

    positive_rate = (
        positive_count / sample_count * 100
        if sample_count > 0
        else 0.0
    )

    is_leaf = left_child == right_child

    if is_leaf:
        leaf_count += 1

        if positive_count > negative_count:
            predicted_class = 1
            answer = "답: 뇌졸중 (1)"
            fill_color = "#FCE4E4"
        else:
            predicted_class = 0
            answer = "답: 아님 (0)"
            fill_color = "#DDEBFA"
            negative_leaf_count += 1

        label = (
            f"{answer}\\n"
            f"훈련 인원: {sample_count}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%"
        )

    else:
        feature_index = int(tree_structure.feature[node_index])
        feature_column = selected_features[feature_index]
        feature_name = FEATURE_NAMES[feature_column]
        threshold = tree_structure.threshold[node_index]

        used_feature_indices.append(feature_index)

        label = (
            f"{feature_name} ≤ {threshold:.2f}?\\n"
            f"훈련 인원: {sample_count}명\\n"
            f"실제 뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_rate:.1f}%"
        )
        fill_color = "#FFF2CC"

    # 노드 번호는 내부 ID로만 사용하고, 화면에는 한국어 설명을 표시합니다.
    safe_label = label.replace('"', '\\"')

    dot_lines.append(
        f'n{node_index} [label="{safe_label}", fillcolor="{fill_color}"];'
    )

    if not is_leaf:
        # 왼쪽 가지: 조건을 만족함(예), 오른쪽 가지: 만족하지 않음(아니요)
        dot_lines.append(
            f'n{node_index} -> n{left_child} [label="예"];'
        )
        dot_lines.append(
            f'n{node_index} -> n{right_child} [label="아니요"];'
        )

dot_lines.append("}")
tree_dot = "\n".join(dot_lines)

st.graphviz_chart(tree_dot, use_container_width=True)


# ==========================================
# 11. 트리 구조 요약
# ==========================================

st.subheader("의사결정트리 요약")

used_features = []

for feature_index in used_feature_indices:
    feature_column = selected_features[feature_index]
    feature_name = FEATURE_NAMES[feature_column]

    if feature_name not in used_features:
        used_features.append(feature_name)

if used_features:
    used_features_text = ", ".join(used_features)
else:
    used_features_text = "질문에 사용한 속성이 없습니다."

st.markdown(
    f"- **답을 내는 마디(잎)의 수:** {leaf_count}개"
)

st.markdown(
    f"- **그중 '아님(0)'이라고 답하는 마디:** "
    f"{negative_leaf_count}개"
)

st.markdown(
    f"- **이 나무가 실제로 질문한 속성:** {used_features_text}"
)


# ==========================================
# 12. 마무리 안내
# ==========================================

st.divider()

st.caption(
    "이 페이지는 분류 모델의 학습과 평가를 연습하기 위한 교육용 도구입니다. "
    "정확도는 데이터 구성과 선택한 속성에 따라 달라질 수 있으며, "
    "개인의 건강 상태를 진단하거나 실제 뇌졸중 발생 가능성을 판단하는 "
    "의료 도구가 아닙니다."
)
