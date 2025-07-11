from django.urls import path
from .views import predict_churn, predict_risk

urlpatterns = [
    path('predict-churn/', predict_churn),
    path('predict-risk/', predict_risk),
]
