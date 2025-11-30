import streamlit as st
import pandas as pd
import json
from model import IncomeModel
from pathlib import Path
from recommendations import get_recommendations, get_recommendations_debug

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "feature_mapping.json"
# --- Load model ---
model = IncomeModel()

# --- Load feature mapping (optional) ---
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        FEATURE_MAP = json.load(f)
except Exception:
    FEATURE_MAP = {}

# --- App config ---
st.set_page_config(page_title="Bank Income Predictor", layout="wide")
st.title(" Прогноз дохода клиента + рекомендации")

# --- Sidebar: client selection ---
st.sidebar.header("Выбор клиента")
uploaded = st.sidebar.file_uploader("CSV клиентов", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    client_id = st.sidebar.selectbox("Client ID", df["client_id"].unique())
    client_row = df[df["client_id"] == client_id]
else:
    st.warning("Загрузите CSV для работы.")
    st.stop()

# --- Predict income ---
st.subheader(" Прогноз дохода")
pred = model.predict(client_row)
ci_low = pred * 0.9
ci_high = pred * 1.1

st.metric("Ожидаемый доход", f"{pred:,.0f} ₽", f"ДИ {ci_low:,.0f} – {ci_high:,.0f}")

# --- SHAP explanation ---
st.subheader(" Объяснение модели (SHAP)")
shap_vals = model.get_shap_values(client_row)
features = client_row.columns
vals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals

shap_df = pd.DataFrame({
    "feature": features,
    "value": client_row.iloc[0].values,
    "shap": vals
}).sort_values("shap", ascending=False).head(5)

st.dataframe(shap_df)

# --- Recommendations ---
st.subheader("Персональные рекомендации")
shap_list = shap_df.to_dict(orient="records")
recs = get_recommendations(int(client_id), pred, shap_list)
st.write(recs)

# --- What-if analysis ---
st.subheader(" What-if анализ")
wa_exp = st.selectbox("Показатель", shap_df["feature"].tolist())
orig = float(client_row[wa_exp].iloc[0])
new = st.slider("Новое значение", orig * 0.5, orig * 1.5, orig)

temp = client_row.copy()
temp[wa_exp] = new
new_pred = model.predict(temp)
st.metric("Новый доход", f"{new_pred:,.0f} ₽", f"Δ {new_pred - pred:,.0f}")

# --- Segment dashboard (simple) ---
st.subheader(" Сегменты клиентов")
seg_counts = df.apply(lambda r: model.predict(r.to_frame().T), axis=1)
seg_df = pd.DataFrame({"income": seg_counts})
seg_df["segment"] = seg_df["income"].apply(lambda x: "high" if x>100000 else "mid" if x>60000 else "low")
st.bar_chart(seg_df["segment"].value_counts())
