import time
from http import HTTPStatus
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from segment import get_segmenter
from recommendations import get_recommendations

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FEATURES = BASE_DIR / "data" / "X_test_final.parquet"
DATA_IDS = BASE_DIR / "data" / "X_test_final_ids.csv"

app = FastAPI(
    title="IncomePrediction API",
    description="Прогнозирование дохода",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Глобальные объекты
X_test: Optional[pd.DataFrame] = None
X_test_ids: Optional[pd.DataFrame] = None
segmenter = None
app_startup_time = time.time()


class ClientRequest(BaseModel):
    client_id: int


class WhatIfRequest(BaseModel):
    client_id: int
    feature_changes: Dict[str, float] = Field(default={})


class PredictResponse(BaseModel):
    client_id: int
    income_prediction: float
    individual_predictions: Dict[str, float]
    shap_explanation: List[Dict[str, Any]]
    model_quality: str
    segment: str
    recommendations: List[str]


class WhatIfResponse(BaseModel):
    client_id: int
    new_income_prediction: float
    delta_income: float
    changes_applied: Dict[str, float]


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


def ensure_ids():

    global X_test, X_test_ids
    if X_test is None:
        X_test = pd.DataFrame()

    if "id" not in X_test.columns:
        # если id нет — создаём по индексу
        X_test["id"] = range(len(X_test))

    if X_test_ids is None or "id" not in X_test_ids.columns or len(X_test_ids) != len(X_test):
        X_test_ids = pd.DataFrame({"id": X_test["id"].values})


def get_client_row(client_id: int) -> Optional[pd.DataFrame]:
    """Возвращает строку DataFrame с одной строкой по client_id."""
    ensure_ids()
    idx_list = X_test_ids.index[X_test_ids["id"] == client_id].tolist()
    if not idx_list:
        return None
    idx = idx_list[0]
    return X_test.iloc[[idx]].copy()


def dummy_prediction(client_row: pd.DataFrame) -> Dict[str, Any]:
    """синтетика"""
    pred_value = 123_456.0
    features = list(client_row.columns)

    return {
        "income_prediction": pred_value,
        "individual_predictions": {
            "catboost": pred_value * 0.9,
            "lightgbm": pred_value * 1.05,
            "xgboost": pred_value * 1.02,
        },
        "shap_explanation": [
            {
                "feature": f,
                "shap_value": 0.0,
                "importance": 0.2
            }
            for f in features[:5]
        ],
        "model_quality": "WMAE: 0.0000",
    }


@app.get("/health/", status_code=HTTPStatus.OK)
def health_check():
    ensure_ids()
    uptime = time.time() - app_startup_time
    return {
        "status": "healthy",
        "model_version": "stub_v1.0",
        "uptime": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m",
        "clients_loaded": len(X_test_ids) if X_test_ids is not None else 0,
        "segments_trained": segmenter is not None and getattr(segmenter, "is_fitted", False),
        "wmae_validation": 0.0,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_client(client_data: ClientRequest):
    client_row = get_client_row(client_data.client_id)
    if client_row is None:
        raise HTTPException(404, f"Client {client_data.client_id} not found")

    pred = dummy_prediction(client_row)
    seg = "unknown"
    if segmenter is not None:
        try:
            seg = segmenter.get_client_segment(client_data.client_id, X_test)
        except Exception:
            seg = "unknown"

    recs = get_recommendations(
        client_id=client_data.client_id,
        income_prediction=pred["income_prediction"],
        shap_explanation=pred["shap_explanation"],
        segment=seg,
    )

    return PredictResponse(
        client_id=client_data.client_id,
        income_prediction=pred["income_prediction"],
        individual_predictions=pred["individual_predictions"],
        shap_explanation=pred["shap_explanation"],
        model_quality=pred["model_quality"],
        segment=seg,
        recommendations=recs,
    )


@app.post("/what_if", response_model=WhatIfResponse)
def what_if_analysis(request: WhatIfRequest):
    client_row = get_client_row(request.client_id)
    if client_row is None:
        raise HTTPException(404, f"Client {request.client_id} not found")

    base_pred = 123_456.0
    delta = 5_000.0
    return WhatIfResponse(
        client_id=request.client_id,
        new_income_prediction=base_pred + delta,
        delta_income=delta,
        changes_applied=request.feature_changes,
    )


@app.get("/segments")
def get_segments():
    ensure_ids()
    if segmenter is None:
        return {"segments": {"stable_high_income": 2, "medium_income": 2, "low": 1}}
    return {"segments": segmenter.get_all_segments(X_test)}


@app.get("/segments/dashboard")
def segments_dashboard():
    ensure_ids()
    if segmenter is None:
        return {
            "distribution": {"stable_high_income": 2, "medium_income": 2, "low": 1},
            "total_clients": 5,
            "feature_importance": {"salary_6to12m_avg": 0.4, "turn_cur_cr_sum_v2": 0.3},
            "thresholds": {"high": 100000, "median": 62754},
        }
    return segmenter.get_segment_dashboard_data(X_test)


@app.on_event("startup")
def startup_event():
    global X_test, X_test_ids, segmenter
    try:
        # Пытаемся загрузить реальные файлы
        X_test = pd.read_parquet(DATA_FEATURES)
        X_test_ids = pd.read_csv(DATA_IDS)
    except Exception:
        # Заглушечные данные
        X_test = pd.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
            }
        )
        X_test_ids = pd.DataFrame({"id": [0, 1, 2, 3, 4]})

    ensure_ids()
    try:
        segmenter = get_segmenter(X_test)
    except Exception:
        segmenter = None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
