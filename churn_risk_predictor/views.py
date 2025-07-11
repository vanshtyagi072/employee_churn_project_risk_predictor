from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib
import pandas as pd
import shap

# Load models and datasets once
employee_model = joblib.load("churn_risk_predictor/employee_model.pkl")
project_model = joblib.load("churn_risk_predictor/project_model.pkl")

raw_df = pd.read_csv("C:/Users/Admin/Desktop/WA_Fn-UseC_-HR-Employee-Attrition.csv")
raw_df['Attrition'] = raw_df['Attrition'].map({'Yes': 1, 'No': 0})
raw_df = raw_df.drop(['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber'], axis=1)
processed_df = pd.get_dummies(raw_df, drop_first=True)
X = processed_df.drop('Attrition', axis=1)

@api_view(['POST'])
def predict_churn(request):
    index = request.data.get('index', 0)
    try:
        index = int(index)
        if index < 0 or index >= len(X):
            return Response({"error": f"Index must be between 0 and {len(X)-1}"}, status=400)

        employee_data = X.iloc[[index]]
        prob = employee_model.predict_proba(employee_data)[0][1]
        pred = employee_model.predict(employee_data)[0]

        if prob > 0.7:
            risk = "🔴 High Risk"
        elif prob > 0.4:
            risk = "🟠 Moderate Risk"
        else:
            risk = "🟢 Low Risk"

        recs = []
        if 'JobSatisfaction' in employee_data.columns and employee_data['JobSatisfaction'].values[0] <= 2:
            recs.append("🧠 Improve job satisfaction.")
        if 'OverTime_Yes' in employee_data.columns and employee_data['OverTime_Yes'].values[0] == 1:
            recs.append("⏰ Reduce overtime.")
        if 'MonthlyIncome' in employee_data.columns and employee_data['MonthlyIncome'].values[0] < raw_df['MonthlyIncome'].median():
            recs.append("💰 Consider salary adjustment.")

        return Response({
            "probability": f"{prob*100:.2f}",
            "risk_level": risk,
            "prediction": "🚨 This employee is at RISK of churn." if pred == 1 else "✅ Likely to stay.",
            "recommendations": recs or ["No major issues."]
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
def predict_risk(request):
    data = request.data
    df_input = pd.DataFrame({
        'Team_Size': [data['team_size']],
        'Project_Budget_USD': [data['budget']],
        'Estimated_Timeline_Months': [data['timeline']],
        'Complexity_Score': [data['complexity']],
        'Stakeholder_Count': [data['stakeholders']],
        'Team_Experience_Level_Junior': [1 if data['experience'] == 'Junior' else 0],
        'Team_Experience_Level_Mixed': [1 if data['experience'] == 'Mixed' else 0],
    })

    # Ensure all expected columns are present
    for col in project_model.feature_names_in_:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[project_model.feature_names_in_]

    # Predict
    pred = project_model.predict(df_input)[0]
    probs = project_model.predict_proba(df_input)[0]

    label = {0: "Low", 1: "Medium", 2: "High"}[pred]

    # --- Smarter Recommendations ---
    recs = []
    if data['complexity'] >= 7:
        recs.append("🔍 Simplify project scope to reduce complexity.")
    if data['timeline'] <= 6 and data['complexity'] >= 6:
        recs.append("⏳ Extend the timeline to accommodate complex tasks.")
    if data['budget'] < 30000 and data['complexity'] >= 6:
        recs.append("💸 Increase budget allocation for high-complexity projects.")
    if data['stakeholders'] > 10:
        recs.append("📢 Improve stakeholder communication strategy.")
    if data['team_size'] < 5:
        recs.append("👥 Add more team members to meet project demands.")
    if data['experience'] == "Junior":
        recs.append("🎓 Provide training or include senior supervision.")

    return Response({
        "label": label,
        "probs": {
            "Low": f"{probs[0]*100:.2f}%",
            "Medium": f"{probs[1]*100:.2f}%",
            "High": f"{probs[2]*100:.2f}%"
        },
        "recommendations": recs or ["✅ No major risk factors."]
    })



def index(request):
    return render(request, 'index.html')


# Create your views here.
