import numpy as np
import pandas as pd


def generate_student_data(num_students=200, weeks=12, seed=42):
    np.random.seed(seed)

    data = []

    for student_id in range(num_students):

        base_ability = np.random.normal(0.6, 0.1)

        for week in range(1, weeks + 1):

            attendance = np.clip(
                np.random.normal(75 + base_ability * 15, 8), 40, 100
            )

            study_hours = np.clip(
                np.random.normal(2 + base_ability * 3, 1), 0, 10
            )

            quiz_score = np.clip(
                np.random.normal(55 + base_ability * 30, 12), 0, 100
            )

            assignments_submitted = np.random.choice(
                [0, 1], p=[0.25, 0.75]
            )

            lms_logins = np.random.randint(1, 20)

            sleep_hours = np.clip(
                np.random.normal(6.5, 1), 4, 9
            )

            data.append([
                student_id,
                week,
                attendance,
                study_hours,
                assignments_submitted,
                quiz_score,
                lms_logins,
                sleep_hours
            ])

    df = pd.DataFrame(data, columns=[
        "student_id",
        "week",
        "attendance",
        "study_hours",
        "assignments_submitted",
        "quiz_score",
        "lms_logins",
        "sleep_hours"
    ])

    return df