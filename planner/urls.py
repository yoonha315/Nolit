from django.urls import path
from .views import PersonaView
 
app_name = "planner"
 
urlpatterns = [
    path('persona/', PersonaView.as_view(), name='persona'),  # 그룹 설정
]