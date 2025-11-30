import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

import shap
from preprocessing import preprocess_client_for_model

BASE_DIR = Path(__file__).resolve().parent.parent


class IncomeModel:
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Загрузка ансамбля + создание SHAP-эксплейнера поверх CatBoost."""
        if model_path is None:
            model_path = BASE_DIR / "models" / "final_3models.pkl"
        else:
            model_path = Path(model_path)

        self.artifacts = joblib.load(model_path)

        self.val_wmae: float = self.artifacts["val_wmae"]
        self.num_features: list[str] = self.artifacts["num_features"]

        self.cat_features_idx: list[int] = self.artifacts.get("cat_features_idx", [])


        # Восстановление списка категориальных фичей
        self.cat_features_idx: list[int] = self.artifacts.get("cat_features_idx", [])
        self.cat_features: list[str] = [
            self.num_features[i] for i in self.cat_features_idx if i < len(self.num_features)
        ]

        # Загружаем модели
        self.cat_model = self.artifacts["cat_model"]
        self.lgb_model = self.artifacts["lgb_model"]
        self.xgb_model = self.artifacts["xgb_model"]
        self.meta_model = self.artifacts["meta_model"]


        try:
            self.shap_explainer = shap.TreeExplainer(self.cat_model)
        except Exception as e:
            print(f"[WARNING] SHAP explainer init failed: {e}")
            self.shap_explainer = None

        print(f"Модель загружена: WMAE={self.val_wmae:.4f}")
        print(f"Фичи: {len(self.num_features)} num + {len(self.cat_features)} cat")


    def _prepare_client(self, client_id: int, test_df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка одного клиента."""
        return preprocess_client_for_model(
            client_id,
            test_df,
            self.num_features,
            self.cat_features,
        )


    def _compute_shap_top(self, client_proc: pd.DataFrame, top_k: int = 5):
        """SHAP-топ факторов."""
        if self.shap_explainer is None:
            return []

        shap_vals = self.shap_explainer.shap_values(client_proc)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = shap_vals[0]

        shap_df = pd.DataFrame(
            {
                "feature": client_proc.columns,
                "shap_value": shap_vals,
            }
        )
        shap_df["shap_abs"] = shap_df["shap_value"].abs()

        top_features = (
            shap_df.nlargest(top_k, "shap_abs")[["feature", "shap_value"]]
            .assign(importance=lambda d: d["shap_value"].abs() / d["shap_value"].abs().sum())
            .to_dict("records")
        )
        return top_features

    def predict_client(self, client_proc: pd.DataFrame) -> dict:
        cat_pred = float(self.cat_model.predict(client_proc)[0])
        lgb_pred = float(self.lgb_model.predict(client_proc[self.num_features])[0])
        xgb_pred = float(self.xgb_model.predict(client_proc[self.num_features])[0])

        pred_stack = np.column_stack([[cat_pred], [lgb_pred], [xgb_pred]])
        final_income = float(self.meta_model.predict(pred_stack)[0])

        shap_top = self._compute_shap_top(client_proc, top_k=5)

        return {
            "client_id": client_proc.index[0] if not client_proc.empty else None,
            "income_prediction": final_income,
            "individual_predictions": {
                "catboost": cat_pred,
                "lightgbm": lgb_pred,
                "xgboost": xgb_pred,
            },
            "shap_explanation": shap_top,
            "model_quality": f"WMAE: {self.val_wmae:.4f}",
        }

    def what_if_predict(self, client_proc: pd.DataFrame, feature_changes: dict) -> dict:
        for feat_name, new_value in feature_changes.items():
            if feat_name in client_proc.columns:
                client_proc[feat_name] = new_value

        cat_pred = float(self.cat_model.predict(client_proc, cat_features=self.cat_features)[0])
        lgb_pred = float(self.lgb_model.predict(client_proc[self.num_features])[0])
        xgb_pred = float(self.xgb_model.predict(client_proc[self.num_features])[0])

        pred_stack = np.column_stack([[cat_pred], [lgb_pred], [xgb_pred]])
        final_income = float(self.meta_model.predict(pred_stack)[0])

        return {
            "new_income_prediction": final_income,
            "changes_applied": feature_changes,
        }

model: Optional[IncomeModel] = None


def load_model() -> IncomeModel:
    global model
    model = IncomeModel()
    return model


def get_model() -> IncomeModel:
    global model
    if model is None:
        model = load_model()
    return model
