import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, Optional, Tuple
from logging_config import main_logger as logger


class ClientSegmenter:
    def __init__(self, n_clusters: int = 3):
        """
        3 сегмента по реальной статистике target
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.is_fitted = False

        # Сегментные имена по умолчанию
        self.segment_names = {
            0: "stable_high",
            1: "medium",
            2: "low"
        }

        # FIX: делаем пороги полем класса
        self.INCOME_THRESHOLDS = {
            'high': 100000,
            'median': 62754
        }

        self.feature_weights = {
            'income_stability': 0.40,
            'turnover_activity': 0.30,
            'demographics': 0.20,
            'debt_risk': 0.10
        }

        self.segment_features = []  # будет заполнено при fit()

    def fit(self, X: pd.DataFrame):
        logger.info(" Fitting ClientSegmenter...")

        candidate_features = [
            # Доходная стабильность (40%)
            'salary_6to12m_avg', 'dp_ils_avg_salary_1y', 'salary_stability',
            'first_salary_income', 'has_salary_flag',
            # Активность (30%)
            'turn_cur_cr_sum_v2', 'turn_cur_db_sum_v2', 'hdb_outstand_sum',
            'turn_cur_cr_7avg_avg_v2', 'turn_cur_cr_avg_v2',
            # Демография (20%)
            'age', 'work_experience',
            # Долговая нагрузка (10%)
            'debt_to_turnover'
        ]

        # Фильтрация существующих фичей
        self.segment_features = [f for f in candidate_features if f in X.columns]
        X_seg = X[self.segment_features].fillna(0)

        logger.info(f" Используется {len(self.segment_features)} фич для сегментации")

        X_scaled = self.scaler.fit_transform(X_seg)
        X_weighted = self._apply_feature_weights(X_scaled)

        # KMeans
        self.kmeans.fit(X_weighted)
        self._adjust_clusters_by_income(X)

        self.is_fitted = True
        logger.info("ClientSegmenter обучен успешно")
        return self

    def _apply_feature_weights(self, X_scaled: np.ndarray) -> np.ndarray:
        weights = np.ones(len(self.segment_features))

        for i, feat in enumerate(self.segment_features):
            if any(k in feat for k in ['salary', 'dp_ils']):
                weights[i] *= self.feature_weights['income_stability']
            elif any(k in feat for k in ['turn', 'hdb']):
                weights[i] *= self.feature_weights['turnover_activity']
            elif feat in ['age', 'work_experience']:
                weights[i] *= self.feature_weights['demographics']
            elif 'debt' in feat:
                weights[i] *= self.feature_weights['debt_risk']

        return X_scaled * weights

    def _adjust_clusters_by_income(self, X: pd.DataFrame):
        """Корректировка сегментов по реальному доходу"""
        if 'target' not in X.columns:
            return

        income_data = X['target'].fillna(X['target'].median())
        cluster_labels = self.kmeans.labels_

        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_income = income_data[cluster_mask].median()

            if cluster_income > self.INCOME_THRESHOLDS['high']:
                self.segment_names[i] = "stable_high_income"
            elif cluster_income > self.INCOME_THRESHOLDS['median']:
                self.segment_names[i] = "medium_income"
            else:
                self.segment_names[i] = "low"

    def get_client_segment(self, client_id: int, df: pd.DataFrame) -> str:
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()")

        client_data = df[df['id'] == client_id]
        if client_data.empty:
            logger.warning(f"Клиент {client_id} не найден")
            return "unknown"

        client_proc = client_data[self.segment_features].fillna(0)
        client_scaled = self.scaler.transform(client_proc)
        client_weighted = self._apply_feature_weights(client_scaled)

        cluster = self.kmeans.predict(client_weighted)[0]
        return self.segment_names.get(cluster, "unknown")

    def get_all_segments(self, df: pd.DataFrame) -> Dict[str, int]:
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit()")

        test_seg = df[self.segment_features].fillna(0)
        test_scaled = self.scaler.transform(test_seg)
        test_weighted = self._apply_feature_weights(test_scaled)
        segments = self.kmeans.predict(test_weighted)

        segment_counts = {name: 0 for name in self.segment_names.values()}
        for cluster, name in self.segment_names.items():
            segment_counts[name] = int(np.sum(segments == cluster))

        return segment_counts

    def get_segment_dashboard_data(self, df: pd.DataFrame) -> Dict:
        return {
            'distribution': self.get_all_segments(df),
            'total_clients': df.shape[0],
            'feature_importance': self._get_feature_importance(),
            'thresholds': self.INCOME_THRESHOLDS
        }

    def _get_feature_importance(self) -> Dict[str, float]:
        if not self.segment_features:
            return {}

        weights = {}
        for feat in self.segment_features[:10]:
            if 'salary' in feat:
                weights[feat] = self.feature_weights['income_stability']
            elif 'turn' in feat:
                weights[feat] = self.feature_weights['turnover_activity']
            elif feat in ['age', 'work_experience']:
                weights[feat] = self.feature_weights['demographics']
            else:
                weights[feat] = self.feature_weights['debt_risk']

        return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))

    def save(self, filepath: str):
        joblib.dump({
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'segment_names': self.segment_names,
            'segment_features': self.segment_features,
            'is_fitted': self.is_fitted,
            'income_thresholds': self.INCOME_THRESHOLDS
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        seg = cls()
        seg.scaler = data['scaler']
        seg.kmeans = data['kmeans']
        seg.segment_names = data['segment_names']
        seg.segment_features = data['segment_features']
        seg.is_fitted = data['is_fitted']
        seg.INCOME_THRESHOLDS = data.get('income_thresholds', seg.INCOME_THRESHOLDS)
        return seg


segmenter = None


def get_segmenter(test_df: pd.DataFrame = None):
    global segmenter
    if segmenter is None:
        segmenter = ClientSegmenter()
        if test_df is not None:
            segmenter.fit(test_df)
    return segmenter