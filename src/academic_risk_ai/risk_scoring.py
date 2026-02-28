def compute_risk_scores(model, feature_df):

    X = feature_df.drop(columns=["student_id", "final_result", "risk_score_simulated"])

    probabilities = model.predict_proba(X)[:, 1]

    feature_df["predicted_risk_probability"] = probabilities

    def categorize(p):
        if p < 0.3:
            return "Low"
        elif p < 0.6:
            return "Medium"
        else:
            return "High"

    feature_df["predicted_risk_level"] = feature_df["predicted_risk_probability"].apply(categorize)

    return feature_df