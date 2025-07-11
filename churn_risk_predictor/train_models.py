import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- Employee Churn ---
df = pd.read_csv("C:/Users/Admin/Desktop/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df = df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'churn_risk_predictor/employee_model.pkl')

# --- Project Risk ---
project_df = pd.read_csv("C:/Users/Admin/Downloads/archive (4)/project_risk_raw_dataset.csv")
project_df['Risk_Level'] = project_df['Risk_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
project_df = project_df.dropna(subset=['Risk_Level'])
project_df = project_df.drop(['Project_ID'], axis=1)
project_df = pd.get_dummies(project_df, drop_first=True)

X_proj = project_df.drop('Risk_Level', axis=1)
y_proj = project_df['Risk_Level']
X_train_p, _, y_train_p, _ = train_test_split(X_proj, y_proj, test_size=0.2, random_state=42)

project_model = RandomForestClassifier(n_estimators=100, random_state=42)
project_model.fit(X_train_p, y_train_p)

joblib.dump(project_model, 'churn_risk_predictor/project_model.pkl')

print("✅ Models trained and saved successfully.")
