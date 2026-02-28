import numpy as np


def recommend_intervention(model, student_row, feature_columns):
    """
    Computes minimal required change for each feature
    and ranks interventions by smallest absolute change.
    """

    student_row = student_row[feature_columns]

    weights = model.coef_[0]
    intercept = model.intercept_[0]

    current_logit = np.dot(weights, student_row.values[0]) + intercept

    recommendations = []

    for i, feature_name in enumerate(feature_columns):

        w_i = weights[i]

        # Skip unstable features
        if abs(w_i) < 1e-8:
            continue

        delta = -current_logit / w_i
        current_value = student_row.values[0][i]
        new_value = current_value + delta

        recommendations.append({
            "feature": feature_name,
            "current_value": current_value,
            "required_change": delta,
            "target_value": new_value,
            "absolute_change": abs(delta)
        })

    # Sort by smallest required change
    recommendations.sort(key=lambda x: x["absolute_change"])

    return {
        "current_logit": current_logit,
        "recommendations_ranked": recommendations
    }