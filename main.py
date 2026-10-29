import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 각 열의 의미를 직접 정리하는 실습 공간입니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "y22un/data/refs/heads/main/stroke.csv"
)

# 기록 파일 경로
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "뇌졸중_데이터_기록.json"


# ==========================================
# 2. 기록 불러오기 및 저장하기
# ==========================================
def load_record():
    """저장된 기록이 있으면 불러오고, 없으면 빈 기록을 반환합니다."""
    default_record = {
        "meanings": {},
        "source": ""
    }

    try:
        if RECORD_FILE.exists():
            with RECORD_FILE.open("r", encoding="utf-8") as file:
                saved_record = json.load(file)

            if not isinstance(saved_record, dict):
                return default_record

            meanings = saved_record.get("meanings", {})
            source = saved_record.get("source", "")

            if not isinstance(meanings, dict):
                meanings = {}

            if not isinstance(source, str):
                source = ""

            return {
                "meanings": meanings,
                "source": source
            }

    except (OSError, json.JSONDecodeError):
        st.warning("저장된 기록을 읽지 못했습니다. 입력 후 다시 저장해 주세요.")

    return default_record


def save_record(meanings, source):
    """우리말 뜻과 데이터 출처를 JSON 파일로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "meanings": meanings,
        "source": source
    }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 3. 데이터 불러오기
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    expected_columns = [
        "id",
        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status",
        "stroke"
    ]

    missing_columns = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "데이터에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    return df


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()


# 저장된 사용자 기록 불러오기
saved_record = load_record()
saved_meanings = saved_record["meanings"]
saved_source = saved_record["source"]


# ==========================================
# 4. 데이터 소개
# ==========================================
st.header("📚 데이터 소개")

st.write(
    """
    이 데이터는 사람들의 기본 정보와 건강 관련 특성,
    그리고 `stroke` 값을 담고 있는 CSV 데이터입니다.

    각 행은 한 사람의 기록을 나타내며, 여러 열을 통해
    데이터의 구조와 값의 종류를 살펴볼 수 있습니다.
    """
)

st.info(
    "`stroke`는 데이터에 기록된 뇌졸중 여부를 나타내는 열입니다. "
    "이 자료는 학습과 분석 실습을 위한 데이터이며, "
    개인의 건강 상태를 진단하는 도구가 아닙니다."
)


# ==========================================
# 5. 숫자 카드 네 개
# ==========================================
total_people = len(df)
total_columns = len(df.columns)

stroke_values = pd.to_numeric(
    df["stroke"],
    errors="coerce"
)

stroke_count = int((stroke_values == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric(
        label="👥 전체 사람 수",
        value=f"{total_people:,}명"
    )

with card2:
    st.metric(
        label="📋 열 개수",
        value=f"{total_columns}개"
    )

with card3:
    st.metric(
        label="🧠 stroke가 1인 사람 수",
        value=f"{stroke_count:,}명"
    )

with card4:
    st.metric(
        label="📊 stroke가 1인 비율",
        value=f"{stroke_ratio:.2f}%"
    )


# ==========================================
# 6. 열 이름과 데이터 정보 표
# ==========================================
st.divider()
st.header("🔎 열 이름과 값 살펴보기")

st.write(
    "각 열의 우리말 뜻을 교재에서 찾아 직접 입력해 보세요. "
    "값의 종류와 빈 값 개수는 데이터에서 자동으로 계산합니다."
)


def describe_values(series):
    """열의 자료형과 대표적인 값의 종류를 설명합니다."""
    non_missing = series.dropna()

    if pd.api.types.is_bool_dtype(series):
        return "논리형"

    if pd.api.types.is_numeric_dtype(series):
        if non_missing.empty:
            return "숫자형 · 값 없음"

        unique_values = sorted(non_missing.unique().tolist())

        if len(unique_values) <= 8:
            values_text = ", ".join(
                str(value) for value in unique_values
            )
            return f"숫자형 · 값: {values_text}"

        return f"숫자형 · 고유값 {len(unique_values):,}종류"

    unique_values = non_missing.astype(str).unique().tolist()

    if len(unique_values) == 0:
        return "문자형 · 값 없음"

    if len(unique_values) <= 6:
        return "범주형 · " + ", ".join(unique_values)

    return f"범주형 · 고유값 {len(unique_values):,}종류"


column_info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [
        str(saved_meanings.get(column, ""))
        for column in df.columns
    ],
    "값의 종류": [
        describe_values(df[column])
        for column in df.columns
    ],
    "빈 값 개수": [
        int(df[column].isna().sum())
        for column in df.columns
    ]
})


# ==========================================
# 7. 우리말 뜻 입력 및 저장
# ==========================================
with st.form("data_notes_form"):
    edited_info = st.data_editor(
        column_info,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=[
            "열 이름",
            "값의 종류",
            "빈 값 개수"
        ],
        column_config={
            "열 이름": st.column_config.TextColumn(
                "열 이름",
                width="medium"
            ),
            "우리말 뜻": st.column_config.TextColumn(
                "우리말 뜻",
                help="교재를 참고해 각 열의 뜻을 입력하세요.",
                width="large"
            ),
            "값의 종류": st.column_config.TextColumn(
                "값의 종류",
                width="large"
            ),
            "빈 값 개수": st.column_config.NumberColumn(
                "빈 값 개수",
                format="%d"
            )
        }
    )

    st.subheader("📝 데이터 출처")

    source_text = st.text_area(
        "교재에 적힌 데이터 출처를 입력하세요.",
        value=saved_source,
        placeholder="여기에 데이터 출처를 입력하세요.",
        height=100
    )

    save_button = st.form_submit_button(
        "💾 기록 저장하기",
        use_container_width=True
    )

    if save_button:
        new_meanings = {}

        for _, row in edited_info.iterrows():
            column_name = row["열 이름"]
            meaning = row["우리말 뜻"]

            new_meanings[column_name] = (
                str(meaning) if pd.notna(meaning) else ""
            )

        try:
            save_record(new_meanings, source_text)
            st.success(
                f"기록을 저장했습니다! 저장 위치: {RECORD_FILE}"
            )
        except OSError as error:
            st.error(
                "기록 파일을 저장하지 못했습니다. "
                "폴더 쓰기 권한을 확인해 주세요."
            )
            st.exception(error)


# ==========================================
# 8. 데이터 첫 다섯 줄
# ==========================================
st.divider()
st.header("👀 데이터 첫 다섯 줄")

st.write(
    "아래는 데이터를 불러온 직후의 첫 다섯 행입니다. "
    "빈 값도 임의로 채우거나 삭제하지 않고 원본 그대로 표시합니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 9. 데이터 출처 안내
# ==========================================
st.divider()
st.header("🔗 데이터 출처")

st.write("**데이터 파일 URL**")
st.code(DATA_URL, language="text")

st.write(
    "교재에 안내된 출처를 위의 입력란에 작성한 뒤 "
    "기록 저장하기를 눌러 주세요."
)

st.caption(
    "저장 파일은 현재 실행 환경의 내_기록 폴더에 생성됩니다. "
    "호스팅 환경이 초기화되거나 파일 시스템이 교체되면 "
    "로컬 JSON 파일이 유지되지 않을 수 있습니다."
)

