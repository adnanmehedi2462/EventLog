from rest_framework import serializers
from .models import *
from django.contrib.auth.models import User

# class EventLogSerializer(serializers.ModelSerializer):
#     user = serializers.PrimaryKeyRelatedField(read_only=True)

#     class Meta:
#         model = EventLog
#         fields = '__all__'

# class RegisterSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = User
#         fields = ('username', 'password', 'email', 'first_name', 'last_name')
#         extra_kwargs = {
#             'password': {'write_only': True},
#             'email': {'required': True},
#         }

#     def create(self, validated_data):
#         user = User.objects.create_user(**validated_data)
#         return user

# class LoginSerializer(serializers.Serializer):
#     username = serializers.CharField(max_length=150)
#     password = serializers.CharField(max_length=128, write_only=True)




# serializers.py



class UploadedCSVSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedCSV
        fields = ('id', 'user', 'csv_file', 'uploaded_at')





from rest_framework import serializers
from .models import EventCSV

class EventCSVSerializer(serializers.ModelSerializer):
    class Meta:
        model = EventCSV
        fields = ['id', 'file', 'uploaded_at']