from django.urls import path
# from .views import EventLogListCreate

from django.contrib.auth.models import User

from .views import EventCSVCreateView
from . import views
app_name = 'mainapp'
urlpatterns = [
    # path('event-logs/', EventLogListCreate.as_view(), name='event-log-list-create'),
    # path('register/', RegisterView.as_view(), name='register'),
    # path('login/', LoginView.as_view(), name='login'),
    # path('upload-csv/', UploadCSVView.as_view(), name='upload_csv'),
    path('eventslog/', EventCSVCreateView.as_view(), name='eventslog-create'),
    path('eventslog/<int:pk>/', EventCSVCreateView.as_view(), name='eventslog-delete'),
    path('admin/view-csv/<int:eventcsv_id>/', views.view_csv_content, name='admin_view_csv'),
    path('admin/view-csv/<int:eventcsv_id>/filter/<str:event_id>/', views.view_csv_content, name='admin_view_csv_filtered'),
    

    


]
