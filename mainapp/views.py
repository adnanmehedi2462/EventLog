# from rest_framework import generics
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.authtoken.views import ObtainAuthToken
# from rest_framework.authtoken.models import Token
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from django.contrib.auth.models import User
# from .models import EventLog
# from .serializers import EventLogSerializer

# from rest_framework import generics, status
# from rest_framework.permissions import IsAuthenticated
# from rest_framework.response import Response
# from .models import EventLog
# from .serializers import EventLogSerializer

# class EventLogListCreate(generics.ListCreateAPIView):
#     serializer_class = EventLogSerializer
#     permission_classes = [IsAuthenticated]

#     def get_queryset(self):
#         return EventLog.objects.filter(user=self.request.user)

#     def perform_create(self, serializer):
#         serializer.save(user=self.request.user)

# from rest_framework import generics, status
# from rest_framework.response import Response
# from rest_framework_simplejwt.tokens import RefreshToken
# from django.contrib.auth import authenticate
# from django.contrib.auth.models import User
# from .serializers import LoginSerializer

# class LoginView(generics.GenericAPIView):
#     serializer_class = LoginSerializer

#     def post(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)

#         # Authenticate user
#         user = authenticate(
#             request,
#             username=serializer.validated_data['username'],
#             password=serializer.validated_data['password']
#         )

#         if user is not None:
#             # Generate tokens
#             refresh = RefreshToken.for_user(user)
#             access_token = str(refresh.access_token)

#             # Return tokens and user data
#             return Response({
#                 'access_token': access_token,
#                 'user': {
#                     'id': user.id,
#                     'username': user.username,
#                     'email': user.email,
#                     'first_name': user.first_name,
#                     'last_name': user.last_name,
#                 }
#             }, status=status.HTTP_200_OK)
#         else:
#             return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
# from rest_framework import generics, status
# from rest_framework.response import Response
# from django.contrib.auth.models import User
# from .serializers import RegisterSerializer

# class RegisterView(generics.CreateAPIView):
#     serializer_class = RegisterSerializer

#     def post(self, request, *args, **kwargs):
#         serializer = self.get_serializer(data=request.data)
#         serializer.is_valid(raise_exception=True)
#         user = serializer.save()
#         return Response({'message': 'User registered successfully'}, status=status.HTTP_201_CREATED)


# from rest_framework import status
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from .models import UploadedCSV
# from .serializers import UploadedCSVSerializer
# from rest_framework.permissions import IsAuthenticated

# class UploadCSVView(APIView):
#     def post(self, request, format=None):
#         serializer = UploadedCSVSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save(user=request.user)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    




# from rest_framework import generics
# from .models import EventCSV
# from .serializers import EventCSVSerializer

# class EventCSVCreateView(generics.ListCreateAPIView):
#     queryset = EventCSV.objects.all()
#     serializer_class = EventCSVSerializer

#     def perform_create(self, serializer):
#         # Delete all existing instances
#         EventCSV.objects.all().delete()
#         serializer.save()



from django.shortcuts import render
import os
from django.conf import settings
from rest_framework import generics
from .models import EventCSV
from .serializers import EventCSVSerializer
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from .models import EventCSV
import csv
from collections import Counter



class EventCSVCreateView(generics.ListCreateAPIView):
    queryset = EventCSV.objects.all()
    serializer_class = EventCSVSerializer

    def perform_create(self, serializer):
        # Delete all existing instances and their associated files
        for old_instance in EventCSV.objects.all():
            if old_instance.file:
                old_file_path = os.path.join(settings.MEDIA_ROOT, old_instance.file.name)
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
            old_instance.delete()
        serializer.save()


def view_csv_content(request, eventcsv_id):
    event_csv = get_object_or_404(EventCSV, pk=eventcsv_id)
    csv_data = []
    total_records = 0
    event_counts = Counter()

    if event_csv.file:
        try:
            event_csv.file.open(mode='r')
            reader = csv.DictReader(event_csv.file)

            filters = {
                'EventID': request.GET.get('EventID'),
                'EventType': request.GET.get('EventType'),
                'SourceName': request.GET.get('SourceName')
            }
            filters = {k: v for k, v in filters.items() if v}

            search_value = request.GET.get('search[value]', '')
            page_size = int(request.GET.get('length', 10))
            page = int(request.GET.get('start', 0)) // page_size + 1

            filtered_data = [
                row for row in reader 
                if all(row.get(k) == v for k, v in filters.items())
                and search_value.lower() in str(row).lower()
            ]
            total_records = len(filtered_data)

            # Aggregate data for the chart
            for row in filtered_data:
                event_counts[row['EventID']] += 1

            # Implement pagination
            start_index = (page - 1) * page_size
            end_index = page * page_size

            csv_data = filtered_data[start_index:end_index]

            event_csv.file.close()
        except Exception as e:
            csv_data = [{'Error': 'Error reading file: {}'.format(e)}]

    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return JsonResponse({
            'data': csv_data,
            'recordsTotal': total_records,
            'recordsFiltered': total_records,
            'event_counts': dict(event_counts)
        }, safe=False)

    return render(request, 'admin/view_csv_chart.html', {'csv_data': csv_data})
