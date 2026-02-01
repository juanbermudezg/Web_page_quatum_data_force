from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.inicio, name='inicio'),
    path('uso-ia/', views.uso_ia, name='uso_ia'),
    path("chatbot/", views.chatbot_page, name="chatbot"),
    path("api/chat/", views.chat_api, name="chat_api"),
    path("energia/", views.consumo_energia, name="consumo_energia"),
]
