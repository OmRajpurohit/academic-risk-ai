import streamlit as st

st.set_page_config(
    page_title="Academic Risk Dashboard",
    layout="wide"
)
import pandas as pd

from academic_risk_ai.data_generator import generate_student_data
from academic_risk_ai.feature_engineering import aggregate_student_features, generate_labels
from academic_risk_ai.train import train_models
from academic_risk_ai.risk_scoring import compute_risk_scores
from academic_risk_ai.intervention import recommend_intervention
from academic_risk_ai.explain import explain_student

@st.cache_resource
def load_system():
    df = generate_student_data(num_students=200, weeks=12)
    features = aggregate_student_features(df)
    features = generate_labels(features)
    best_model, feature_columns = train_models(features)
    return features, best_model, feature_columns

features, best_model, feature_columns = load_system()


#teacher friendly labels
FEATURE_LABELS = {
    "quiz_trend": "Quiz Performance Trend",
    "assignments_missed": "Missed Assignments",
    "attendance_mean": "Average Attendance",
    "attendance_trend": "Attendance Trend",
    "study_hours_mean": "Average Study Hours",
    "sleep_mean": "Average Sleep Hours",
    "lms_mean": "LMS Engagement",
    "study_trend": "Study Time Trend",
    "quiz_mean": "Average Quiz Score"
}

# for UI
CONTROLLABLE_FEATURES = [
    "quiz_trend",
    "assignments_missed",
    "attendance_mean",
    "study_hours_mean"
]

st.title("🎓 Academic Risk Dashboard")
st.caption("AI-powered early warning and intervention support")



# Generate synthetic data
df = generate_student_data(num_students=200, weeks=12)
features = aggregate_student_features(df)
features = generate_labels(features)

best_model, feature_columns = train_models(features)

scored_df = compute_risk_scores(best_model, features)

# Student selection
student_ids = scored_df["student_id"].tolist()
selected_student = st.selectbox("Select Student ID", student_ids)

student_row = scored_df[scored_df["student_id"] == selected_student]

X = student_row[feature_columns]

prob = best_model.predict_proba(X)[0][1]

st.subheader("📊 Risk Prediction")
st.metric("Failure Probability", f"{prob:.2%}")

if prob > 0.6:
    st.error("High Risk")
elif prob > 0.3:
    st.warning("Medium Risk")
else:
    st.success("Low Risk")

    

# SHAP explanation
st.subheader("🔍 Explanation")
from academic_risk_ai.explain import explain_student_plain

# st.subheader("📌 Why Is This Student At Risk?")

with st.expander("📌 Why Is This Student At Risk? (Click to Expand)"):

    explanation = explain_student_plain(
        best_model,
        features[feature_columns],
        feature_columns,
        student_index=selected_student
    )

    st.write("### 🔴 Main Risk Drivers")

    for f, val in explanation["top_risk_factors"]:
        st.write(f"- {FEATURE_LABELS.get(f, f)}")

    st.write("### 🟢 Protective Factors")

    for f, val in explanation["top_protective_factors"]:
        st.write(f"- {FEATURE_LABELS.get(f, f)}")


#Visual feature Overview
with st.expander("📊 Student Performance Breakdown (Click to Expand)"):

    import matplotlib.pyplot as plt

    student_data = student_row[feature_columns].iloc[0]

    fig, ax = plt.subplots()
    ax.barh(
        [FEATURE_LABELS.get(f, f) for f in feature_columns],
        student_data.values
    )
    ax.set_xlabel("Value")

    st.pyplot(fig)


#interactive simulation slider
st.subheader("🛠 Try Intervention Simulation")
st.subheader("🛠 Intervention Simulator")

col1, col2 = st.columns(2)

modified_row = student_row[feature_columns].copy()

for i, feature in enumerate(CONTROLLABLE_FEATURES):

    if i % 2 == 0:
        container = col1
    else:
        container = col2

    with container:
        modified_row[feature] = st.slider(
            FEATURE_LABELS.get(feature, feature),
            float(features[feature].min()),
            float(features[feature].max()),
            float(modified_row[feature].values[0])
        )

new_prob = best_model.predict_proba(modified_row)[0][1]

st.metric("Updated Risk Probability", f"{new_prob:.2%}")

# Recommendation
st.subheader("💡 Recommended Intervention")

recommendation = recommend_intervention(best_model, X, feature_columns)

top_rec = recommendation["recommendations_ranked"][0]

st.write(f"**Best Feature to Improve:** {top_rec['feature']}")
st.write(f"Required Change: {top_rec['required_change']:.3f}")
st.write(f"Target Value: {top_rec['target_value']:.3f}")