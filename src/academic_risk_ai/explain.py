import shap
import matplotlib.pyplot as plt

def explain_student(model, X, student_index=0):

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[student_index], show=False)

    return fig

def explain_student_plain(model, X, feature_columns, student_index=0):
    import shap
    import numpy as np

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer(X)

    values = shap_values.values[student_index]

    feature_contrib = list(zip(feature_columns, values))

    # Sort by contribution
    feature_contrib.sort(key=lambda x: x[1], reverse=True)

    top_risk = [f for f in feature_contrib if f[1] > 0][:3]
    top_protective = [f for f in feature_contrib if f[1] < 0][:3]

    return {
        "top_risk_factors": top_risk,
        "top_protective_factors": top_protective
    }