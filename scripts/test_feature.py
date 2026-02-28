from academic_risk_ai.data_generator import generate_student_data
from academic_risk_ai.feature_engineering import aggregate_student_features, generate_labels

df = generate_student_data(num_students=200, weeks=12)

features = aggregate_student_features(df)

features = generate_labels(features)

print(features.head())
print(features["final_result"].value_counts())