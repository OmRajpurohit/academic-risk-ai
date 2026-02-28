from academic_risk_ai.data_generator import generate_student_data

df = generate_student_data(num_students=5, weeks=4)

print(df.head())
print(df.shape)