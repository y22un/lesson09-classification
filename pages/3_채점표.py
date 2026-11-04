import streamlit as st
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier


# ==========================================
# 1. 페이지 설정
# ==========================================

st.title("📋 채점표")
st.caption(
    "테스트 데이터로 모델의 분류 결과를 채점하고, "
    "놓친 사람과 잘못 판단한 사람을 확인합니다."
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

MODEL_LOGISTIC = "로지스틱 회귀(확률로 답하는 모델)"
MODEL_TREE = "의사결정트리(질문으로 답하는 모델)"
MODEL_BASELINE = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

RANDOM_STATE = 42


# ==========================================
# 2. 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    """첫 화면과 같은 주소에서 UTF-8로 데이터를 불러옵니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()
except Exception as error:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "CSV 주소와 인터넷 연결을 확인해 주세요."
    )
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

for column in FEATURE_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df.empty:
    st.error("분석할 데이터가 없습니다.")
    st.stop()


# ==========================================
# 3. 입력 속성 선택
# ==========================================

st.header("1. 분류에 사용할 속성")

st.write(
    "분류 모델 페이지와 마찬가지로 입력 속성을 선택할 수 있습니다. "
    "처음에는 체질량지수를 제외한 네 가지 속성이 선택되어 있습니다."
)

default_features = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

selected_features = st.multiselect(
    "입력 속성",
    options=FEATURE_COLUMNS,
    default=default_features,
    format_func=lambda column: FEATURE_NAMES[column],
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()


# ==========================================
# 4. 학습용과 테스트용 분리
# ==========================================

st.header("2. 테스트 데이터 확인")

# 분류 모델 페이지와 동일한 방식:
# 번호순으로 정렬한 뒤 10명마다 앞의 3명을 테스트용으로 사용합니다.
test_mask = pd.Series(
    [position % 10 < 3 for position in range(len(df))]
)

X = df[selected_features].copy()
y = df["stroke"].copy()

X_train = X.loc[~test_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[~test_mask].copy()
y_test = y.loc[test_mask].copy()

test_positive_count = int(y_test.eq(1).sum())

st.write(
    f"**테스트용 사람 수: {len(y_test):,}명**  ·  "
    f"**그중 실제 뇌졸중인 사람 수: {test_positive_count:,}명**"
)

if len(y_test) != 1533:
    st.warning(
        f"현재 테스트용 데이터는 {len(y_test):,}명입니다. "
        "요청한 1,533명과 다르다면 CSV의 행 수와 데이터 전처리를 확인해 주세요."
    )


# ==========================================
# 5. 동일한 조건으로 모델 학습
# ==========================================

st.header("3. 모델 학습 및 채점")

# BMI가 선택된 경우에만 중앙값 대체를 수행합니다.
# 중앙값은 학습용 데이터로만 계산합니다.
logistic_steps = []

if "bmi" in selected_features:
    logistic_steps.append(
        ("imputer", SimpleImputer(strategy="median"))
    )

# 크기 조정 역시 학습용 데이터로만 계산합니다.
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


# 기준 모델은 입력 속성을 보지 않고 훈련용에서 다수인 클래스만 예측합니다.
baseline_model = DummyClassifier(
    strategy="most_frequent",
    random_state=RANDOM_STATE,
)

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


# 각 모델의 훈련용 및 테스트용 예측
models = {
    MODEL_LOGISTIC: logistic_model,
    MODEL_TREE: tree_model,
    MODEL_BASELINE: baseline_model,
}

predictions = {
    MODEL_LOGISTIC: {
        "train": logistic_model.predict(X_train),
        "test": logistic_model.predict(X_test),
    },
    MODEL_TREE: {
        "train": tree_model.predict(X_train),
        "test": tree_model.predict(X_test),
    },
    MODEL_BASELINE: {
        "train": baseline_model.predict(X_train_baseline),
        "test": baseline_model.predict(X_test_baseline),
    },
}


# ==========================================
# 6. 혼동행렬과 지표 계산 함수
# ==========================================

def calculate_confusion(y_true, y_pred):
    """
    양성은 stroke=1입니다.
    TP: 실제 1, 예측 1
    FP: 실제 0, 예측 1
    FN: 실제 1, 예측 0
    TN: 실제 0, 예측 0
    """
    actual = pd.Series(y_true).reset_index(drop=True).astype(int)
    predicted = pd.Series(y_pred).reset_index(drop=True).astype(int)

    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "actual_0": tn + fp,
        "actual_1": tp + fn,
        "predicted_0": tn + fn,
        "predicted_1": tp + fp,
        "total": tp + fp + fn + tn,
    }


def calculate_metrics(counts):
    """혼동행렬의 네 칸으로 네 가지 지표를 계산합니다."""
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    tn = counts["TN"]
    total = counts["total"]

    accuracy = (tp + tn) / total if total > 0 else None

    actual_positive = tp + fn
    predicted_positive = tp + fp

    recall = (
        tp / actual_positive
        if actual_positive > 0
        else None
    )

    # 양성 예측이 0명이면 정밀도와 F1은 계산할 수 없습니다.
    if predicted_positive == 0:
        precision = None
        f1 = None
    else:
        precision = tp / predicted_positive

        if recall is None:
            f1 = None
        elif precision + recall == 0:
            f1 = 0.0
        else:
            f1 = (
                2 * precision * recall
                / (precision + recall)
            )

    return {
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
    }


def format_metric(value):
    """지표를 백분율로 표시합니다."""
    if value is None:
        return "계산할 수 없음"
    return f"{value:.2%}"


test_counts = {}
test_metrics = {}

for model_name in models:
    test_counts[model_name] = calculate_confusion(
        y_test,
        predictions[model_name]["test"],
    )
    test_metrics[model_name] = calculate_metrics(
        test_counts[model_name]
    )


# ==========================================
# 7. 혼동행렬 표시 함수
# ==========================================

def show_confusion_matrix(model_name, counts):
    """행은 실제, 열은 예측인 2×2 혼동행렬과 합계를 표시합니다."""
    st.subheader(model_name)

    matrix = pd.DataFrame(
        [
            [
                f"TN: {counts['TN']:,}명",
                f"FP: {counts['FP']:,}명",
                f"{counts['actual_0']:,}명",
            ],
            [
                f"FN: {counts['FN']:,}명",
                f"TP: {counts['TP']:,}명",
                f"{counts['actual_1']:,}명",
            ],
            [
                f"{counts['predicted_0']:,}명",
                f"{counts['predicted_1']:,}명",
                f"{counts['total']:,}명",
            ],
        ],
        index=[
            "실제: 아님 (0)",
            "실제: 뇌졸중 (1)",
            "열 합계",
        ],
        columns=[
            "예측: 아님 (0)",
            "예측: 뇌졸중 (1)",
            "행 합계",
        ],
    )

    st.dataframe(
        matrix,
        use_container_width=True,
    )

    st.caption(
        "TN=아님을 맞힘 · FP=아님인데 뇌졸중이라고 예측 · "
        "FN=뇌졸중인데 아님이라고 예측 · TP=뇌졸중을 맞힘"
    )


# ==========================================
# 8. 두 모델의 혼동행렬
# ==========================================

st.header("4. 테스트 데이터 혼동행렬")

left_col, right_col = st.columns(2)

with left_col:
    show_confusion_matrix(
        MODEL_LOGISTIC,
        test_counts[MODEL_LOGISTIC],
    )

with right_col:
    show_confusion_matrix(
        MODEL_TREE,
        test_counts[MODEL_TREE],
    )


# ==========================================
# 9. 네 가지 지표 비교표
# ==========================================

st.header("5. 정확도 · 재현율 · 정밀도 · F1 비교")

st.write(
    "- **정확도:** 전체 테스트 사람 중 맞힌 비율\n"
    "- **재현율:** 실제 뇌졸중인 사람 중 뇌졸중이라고 맞힌 비율\n"
    "- **정밀도:** 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율\n"
    "- **F1:** 정밀도와 재현율을 함께 고려한 값"
)

metric_rows = []

for model_name in [
    MODEL_LOGISTIC,
    MODEL_TREE,
    MODEL_BASELINE,
]:
    if model_name == MODEL_BASELINE:
        counts = calculate_confusion(
            y_test,
            predictions[MODEL_BASELINE]["test"],
        )
        metrics = calculate_metrics(counts)
    else:
        counts = test_counts[model_name]
        metrics = test_metrics[model_name]

    metric_rows.append({
        "모델": model_name,
        "정확도": format_metric(metrics["정확도"]),
        "재현율": format_metric(metrics["재현율"]),
        "정밀도": format_metric(metrics["정밀도"]),
        "F1": format_metric(metrics["F1"]),
    })

metric_table = pd.DataFrame(metric_rows)

st.dataframe(
    metric_table,
    hide_index=True,
    use_container_width=True,
)


# ==========================================
# 10. 놓친 사람과 헛짚은 사람 목록 준비
# ==========================================

st.header("6. 놓친 사람과 헛짚은 사람")

st.write(
    "아래에서 모델을 선택하면 해당 모델의 테스트 결과를 확인할 수 있습니다."
)

selected_model_name = st.selectbox(
    "목록을 확인할 모델",
    options=[
        MODEL_LOGISTIC,
        MODEL_TREE,
        MODEL_BASELINE,
    ],
    key="error_list_model",
)

selected_predictions = predictions[selected_model_name]["test"]

# X_test의 인덱스는 정렬된 원본 df의 행 번호와 같습니다.
error_table = df.loc[
    X_test.index,
    [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
        "stroke",
    ],
].copy()

error_table["예측"] = selected_predictions
error_table["실제"] = y_test.to_numpy()

# FN: 실제 뇌졸중(1), 예측은 아님(0)
missed_people = error_table[
    (error_table["실제"] == 1)
    & (error_table["예측"] == 0)
].copy()

# FP: 실제 아님(0), 예측은 뇌졸중(1)
false_alarm_people = error_table[
    (error_table["실제"] == 0)
    & (error_table["예측"] == 1)
].copy()


def prepare_people_list(people):
    """목록에 필요한 열을 한글로 바꾸고 나이 내림차순으로 정렬합니다."""
    result = people.sort_values(
        "age",
        ascending=False,
        kind="stable",
        na_position="last",
    ).copy()

    result = result.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )

    result["고혈압 여부"] = result["고혈압 여부"].map(
        {0: "없음", 1: "있음"}
    ).fillna("확인 불가")

    result["심장병 여부"] = result["심장병 여부"].map(
        {0: "없음", 1: "있음"}
    ).fillna("확인 불가")

    result["번호"] = result["번호"].map(
        lambda value: (
            str(int(value))
            if pd.notna(value) and float(value).is_integer()
            else str(value)
        )
    )

    return result[
        [
            "번호",
            "나이",
            "평균 혈당",
            "고혈압 여부",
            "심장병 여부",
        ]
    ]


missed_display = prepare_people_list(missed_people)
false_alarm_display = prepare_people_list(false_alarm_people)

missed_total = len(missed_display)
false_alarm_total = len(false_alarm_display)

missed_col, false_alarm_col = st.columns(2)

with missed_col:
    st.subheader(f"놓친 사람 (FN): {missed_total:,}명")
    st.caption(
        "실제로 뇌졸중이 기록되어 있지만 모델은 아님(0)이라고 예측했습니다."
    )

    st.dataframe(
        missed_display.head(20),
        hide_index=True,
        use_container_width=True,
    )

    if missed_total > 20:
        st.caption(
            f"전체 {missed_total:,}명 중 나이가 많은 순서로 "
            "앞의 20명만 표시했습니다."
        )
    elif missed_total == 0:
        st.info("놓친 사람이 없습니다.")

with false_alarm_col:
    st.subheader(f"헛짚은 사람 (FP): {false_alarm_total:,}명")
    st.caption(
        "실제로는 뇌졸중이 아니지만 모델은 뇌졸중(1)이라고 예측했습니다."
    )

    st.dataframe(
        false_alarm_display.head(20),
        hide_index=True,
        use_container_width=True,
    )

    if false_alarm_total > 20:
        st.caption(
            f"전체 {false_alarm_total:,}명 중 나이가 많은 순서로 "
            "앞의 20명만 표시했습니다."
        )
    elif false_alarm_total == 0:
        st.info("헛짚은 사람이 없습니다.")


# ==========================================
# 11. 혼동행렬과 목록 인원 일치 확인
# ==========================================

selected_counts = calculate_confusion(
    y_test,
    selected_predictions,
)

st.info(
    f"목록 인원 확인 — 놓친 사람 {missed_total:,}명 = "
    f"혼동행렬 FN {selected_counts['FN']:,}명, "
    f"헛짚은 사람 {false_alarm_total:,}명 = "
    f"혼동행렬 FP {selected_counts['FP']:,}명."
)

st.caption(
    "이 페이지는 분류 모델을 학습하고 평가하는 교육용 실습입니다. "
    "놓친 사람과 헛짚은 사람은 모델의 예측 오류를 이해하기 위한 "
    "데이터상의 분류이며, 개인의 건강 상태를 진단하는 결과가 아닙니다."
)
