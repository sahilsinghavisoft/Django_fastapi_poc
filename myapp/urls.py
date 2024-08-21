# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('student/register/', views.student_register, name='student_register'),
    path('professor/register/', views.professor_register, name='professor_register'),
    path('student/login/', views.student_login, name='student_login'),
    path('professor/login/', views.professor_login, name='professor_login'),
    path('logout/', views.user_logout, name='logout'),
]
