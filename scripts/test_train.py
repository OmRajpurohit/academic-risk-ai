from academic_risk_ai.data_generator import generate_student_data
from academic_risk_ai.feature_engineering import aggregate_student_features, generate_labels
from academic_risk_ai.train import train_models
from academic_risk_ai.risk_scoring import compute_risk_scores
from academic_risk_ai.model_utils import save_model

df = generate_student_data(num_students=1000, weeks=12)
features = aggregate_student_features(df)
features = generate_labels(features)

best_model = train_models(features)

save_model(best_model)

scored_df = compute_risk_scores(best_model, features)