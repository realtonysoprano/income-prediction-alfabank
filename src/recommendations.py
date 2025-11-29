import pandas as pd
from typing import Dict, Any, List
from logging_config import main_logger as logger

# Пороги по реальной статистике target
INCOME_THRESHOLDS = {
    'high': 100000,  # 75% квантиль
    'median': 62754  # Медиана
}

# Продукты по сегментам (обновленные пороги)
PRODUCTS_BY_SEGMENT = {
    "stable_high_income": [  # >100k (25% топ-клиентов)
        "Ипотека (от 7.5%)",
        "Инвестиции в фонды",
        "Platinum карта (оверлимит 500k)",
        "Овердрафт бизнес"
    ],
    "medium_risky": [  # 62.7k-100k (~25% клиентов)
        "Автокредит (9.9%)",
        "Кредитка Gold (grace 120 дней)",
        "Накопительный счет 12%",
        "Рефинансирование кредитов"
    ],
    "low_stable": [  # <62.7k (48% клиентов)
        "Кредитка Classic (лимит 100k)",
        "Дебетовая Premium + 3% кэшбэк",
        "Микрозайм (до 50k)",
        "Зарплатный пакет"
    ]
}

# SHAP → дополнительные продукты (топ-фичи из model.py)
SHAP_BOOSTERS = {
    "salary_6to12m_avg": "Накопительный счет Premium 12%",
    "salary_stability": "Инвестиции (стабильный доход)",
    "dp_ils_avg_salary_1y": "Зарплатный проект + бонусы",
    "hdb_outstand_sum": "Рефинансирование долгов",
    "turn_cur_cr_sum_v2": "Бизнес-счет для ИП",
    "work_experience": "Кредит на образование/повышение квалификации",
    "debt_to_turnover": "Программа снижения долговой нагрузки",
    "first_salary_income": "Первая зарплата → Welcome бонус",
    "has_salary_flag": "Зарплатный Premium пакет",
    "age": "Пенсионное планирование" if "age" in "age" else None
}

# Правила исключения по сегментам
SEGMENT_EXCLUSIONS = {
    "stable_high_income": ["Микрозайм", "Кредитка Classic"],
    "medium_risky": ["Овердрафт бизнес"],
    "low_stable": ["Ипотека", "Инвестиции", "Platinum карта"]
}


def get_recommendations(client_id: int, income_prediction: float,
                        shap_explanation: list = None,
                        segment: str = None) -> List[str]:
    """
    Главная функция для API /predict

    """
    logger.info(f"Рекомендации для клиента {client_id}: доход={income_prediction:,.0f}₽")

    # 1. Базовый список по сегменту
    base_recs = _get_base_recommendations(income_prediction, segment)

    # 2. SHAP-усиление (топ-3 фичи)
    shap_recs = _apply_shap_boosters(base_recs, shap_explanation)

    # 3. Фильтрация по правилам сегмента
    final_recs = _apply_segment_rules(shap_recs, segment)

    # 4. Топ-3 рекомендации
    recommendations = final_recs[:3]

    logger.info(f"Итог ({len(recommendations)}): {recommendations}")
    return recommendations


def _get_base_recommendations(income: float, segment: str = None) -> List[str]:
    """База по доходу/сегменту"""
    # Приоритет реальному сегменту
    if segment and segment in PRODUCTS_BY_SEGMENT:
        recs = PRODUCTS_BY_SEGMENT[segment][:]
        logger.debug(f"База по сегменту {segment}")
    else:
        # Fallback по доходу
        if income >= INCOME_THRESHOLDS['high']:
            recs = PRODUCTS_BY_SEGMENT['stable_high_income'][:]
        elif income >= INCOME_THRESHOLDS['median']:
            recs = PRODUCTS_BY_SEGMENT['medium_risky'][:]
        else:
            recs = PRODUCTS_BY_SEGMENT['low_stable'][:]
        logger.debug(f"База по доходу {income:,.0f}₽")

    return recs


def _apply_shap_boosters(base_recs: List[str], shap_explanation: list) -> List[str]:
    """SHAP топ-3 доп. продукты"""
    if not shap_explanation:
        return base_recs

    shap_recs = base_recs.copy()

    # Топ-3 SHAP фичи
    for item in shap_explanation[:3]:
        feature = item['feature'].lower()
        if feature in SHAP_BOOSTERS and SHAP_BOOSTERS[feature]:
            shap_recs.append(SHAP_BOOSTERS[feature])

    # Убираем дубликаты
    return list(set(shap_recs))


def _apply_segment_rules(recommendations: List[str], segment: str) -> List[str]:
    """Исключения по сегменту"""
    if not segment or segment not in SEGMENT_EXCLUSIONS:
        return recommendations

    exclusions = SEGMENT_EXCLUSIONS[segment]
    filtered = [r for r in recommendations
                if not any(exclusion.lower() in r.lower() for exclusion in exclusions)]

    logger.debug(f"Исключено по сегменту {segment}: {len(recommendations) - len(filtered)}")
    return filtered


def get_recommendations_debug(client_id: int, income: float,
                              shap_explanation: list, segment: str) -> Dict:
    """Для отладки/логов"""
    recs = get_recommendations(client_id, income, shap_explanation, segment)

    return {
        "client_id": client_id,
        "income": income,
        "income_segment": _get_income_segment(income),
        "ml_segment": segment,
        "shap_top3": [item['feature'] for item in shap_explanation[:3]],
        "final_recommendations": recs,
        "logic_applied": "income - segment - SHAP - exclusions"
    }


def _get_income_segment(income: float) -> str:
    """Чистый income-сегмент"""
    if income >= INCOME_THRESHOLDS['high']:
        return "stable_high_income"
    elif income >= INCOME_THRESHOLDS['median']:
        return "medium_risky"
    return "low_stable"
