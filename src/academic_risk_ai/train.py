import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_models(feature_df):

    X = feature_df.drop(columns=["student_id", "final_result", "risk_score_simulated"])
    y = feature_df["final_result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n===== Logistic Regression =====")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))

    print("\n===== Random Forest =====")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))

    



    print("\nSelecting best model based on test accuracy...")

    lr_score = lr.score(X_test, y_test)
    rf_score = rf.score(X_test, y_test)

    best_model = lr if lr_score > rf_score else rf

    feature_columns = X.columns.tolist()

    print("Best Model:", type(best_model).__name__)

    return best_model, feature_columns