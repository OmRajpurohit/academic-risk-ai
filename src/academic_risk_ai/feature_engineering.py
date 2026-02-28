import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_trend_slope(values):
    """
    Computes slope of a time series using linear regression.
    """
    X = np.arange(len(values)).reshape(-1, 1)
    y = np.array(values)

    model = LinearRegression()
    model.fit(X, y)

    return model.coef_[0]


def aggregate_student_features(df):
    """
    Aggregates weekly data into student-level features.
    """

    student_features = []

    for student_id, group in df.groupby("student_id"):

        group = group.sort_values("week")

        attendance_mean = group["attendance"].mean()
        study_hours_mean = group["study_hours"].mean()
        quiz_mean = group["quiz_score"].mean()
        lms_mean = group["lms_logins"].mean()
        sleep_mean = group["sleep_hours"].mean()

        assignments_missed = (1 - group["assignments_submitted"]).sum()

        quiz_trend = compute_trend_slope(group["quiz_score"])
        attendance_trend = compute_trend_slope(group["attendance"])
        study_trend = compute_trend_slope(group["study_hours"])

        student_features.append([
            student_id,
            attendance_mean,
            study_hours_mean,
            quiz_mean,
            lms_mean,
            sleep_mean,
            assignments_missed,
            quiz_trend,
            attendance_trend,
            study_trend
        ])

    feature_df = pd.DataFrame(student_features, columns=[
        "student_id",
        "attendance_mean",
        "study_hours_mean",
        "quiz_mean",
        "lms_mean",
        "sleep_mean",
        "assignments_missed",
        "quiz_trend",
        "attendance_trend",
        "study_trend"
    ])

    return feature_df


def generate_labels(feature_df):
    """
    Generates pass/fail label based on academic risk factors.
    1 = Fail (High Risk)
    0 = Pass (Low Risk)
    """

    risk_score = (
        0.4 * (70 - feature_df["quiz_mean"]) +
        5 * feature_df["assignments_missed"] +
        0.3 * (75 - feature_df["attendance_mean"]) +
        (-50 * feature_df["quiz_trend"])
    )

    # Normalize risk
    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

    feature_df["final_result"] = (risk_score > 0.5).astype(int)
    feature_df["risk_score_simulated"] = risk_score

    return feature_df