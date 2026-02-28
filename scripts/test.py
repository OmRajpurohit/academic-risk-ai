from academic_risk_ai.data_generator import generate_student_data
from academic_risk_ai.feature_engineering import aggregate_student_features, generate_labels
from academic_risk_ai.train import train_models
from academic_risk_ai.risk_scoring import compute_risk_scores
from academic_risk_ai.model_utils import save_model
from academic_risk_ai.simulate import simulate_feature_change

# 1️⃣ Generate data
df = generate_student_data(num_students=200, weeks=12)

# 2️⃣ Feature engineering
features = aggregate_student_features(df)
features = generate_labels(features)

# 3️⃣ Train model
best_model, feature_columns = train_models(features)

# 4️⃣ Save model
save_model(best_model)

# 5️⃣ Risk scoring
scored_df = compute_risk_scores(best_model, features)

# print(scored_df[[
#     "student_id",
#     "predicted_risk_probability",
#     "predicted_risk_level"
# ]].head())

# # 6️⃣ Simulation

# # Prepare input row (must remove non-model columns)
# X = features.drop(columns=["student_id", "final_result", "risk_score_simulated"])

# sample_student = X.iloc[[0]]  # double brackets keep it 2D

# result = simulate_feature_change(
#     best_model,
#     sample_student,
#     "attendance_mean",
#     95,
#     feature_columns
# )

# print("\nSimulation Result:")
# print(result)

# import pandas as pd

# coef_df = pd.DataFrame({
#     "feature": feature_columns,
#     "coefficient": best_model.coef_[0]
# })

# print(coef_df.sort_values("coefficient", ascending=False))


# import matplotlib.pyplot as plt

# coef_df_sorted = coef_df.sort_values("coefficient")

# plt.figure()
# plt.barh(coef_df_sorted["feature"], coef_df_sorted["coefficient"])
# plt.ylabel("Coefficient Value")
# plt.title("Logistic Regression Feature Importance")
# plt.show()


from academic_risk_ai.explain import explain_student

# Prepare model input matrix
X = features[feature_columns]
# Explain first student
shap_values = explain_student(best_model, X, student_index=0)

print("\nSHAP Values for Student 0:")
print(shap_values)


# from academic_risk_ai.intervention import minimal_change_to_reduce_risk

# X = features[feature_columns]
# sample_student = X.iloc[[0]]

# intervention_result = minimal_change_to_reduce_risk(
#     best_model,
#     sample_student,
#     "quiz_trend",
#     feature_columns
# )

# print("\nIntervention Analysis:")
# print(intervention_result)


from academic_risk_ai.intervention import recommend_intervention

X = features[feature_columns]
sample_student = X.iloc[[0]]

recommendation_output = recommend_intervention(
    best_model,
    sample_student,
    feature_columns
)

print("\nTop 3 Recommended Interventions:")
for rec in recommendation_output["recommendations_ranked"][:3]:
    print(rec)

