def simulate_feature_change(model, student_row, feature_name, new_value, feature_columns):

    modified_row = student_row.copy()
    modified_row[feature_name] = new_value

    # Ensure correct column order
    student_row = student_row[feature_columns]
    modified_row = modified_row[feature_columns]

    original_prob = model.predict_proba(student_row)[0][1]
    new_prob = model.predict_proba(modified_row)[0][1]

    return {
        "original_risk_probability": original_prob,
        "new_risk_probability": new_prob,
        "risk_change": new_prob - original_prob
    }