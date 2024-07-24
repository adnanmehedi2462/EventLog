
from django.urls import path
from .views import EventCSVCreateView, view_csv_content, predict_maintenance, TrainModelView

app_name = 'mainapp'
urlpatterns = [
    path('eventslog/', EventCSVCreateView.as_view(), name='eventslog-create'),
    path('eventslog/<int:pk>/', EventCSVCreateView.as_view(), name='eventslog-delete'),
    path('admin/view-csv/<int:eventcsv_id>/', view_csv_content, name='admin_view_csv'),
    path('admin/view-csv/<int:eventcsv_id>/filter/<str:event_id>/', view_csv_content, name='admin_view_csv_filtered'),
    path('admin/predict/<int:eventcsv_id>/', predict_maintenance, name='predict_maintenance'),
    path('admin/train/<int:pk>/', TrainModelView.as_view(), name='train_model'),
    path('train_model/<int:pk>/', TrainModelView.as_view(), name='train_model'),
    path('api/predict_maintenance/<int:pk>/', predict_maintenance, name='predict_maintenance'),
]