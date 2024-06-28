from django.db import models
from django.contrib.auth.models import User

from django.db import models
from django.contrib.auth.models import User


class EventLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    time_generated = models.TextField(null=True, blank=True)
    event_id = models.TextField(null=True, blank=True)
    event_type = models.TextField(null=True, blank=True)
    event_category = models.TextField(null=True, blank=True)
    source_name = models.CharField(max_length=255, null=True, blank=True)
    computer_name = models.CharField(max_length=255, null=True, blank=True)
    string_inserts = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.time_generated} - {self.source_name} - {self.event_id}"



class UploadedCSV(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='uploaded_csv/')
    uploaded_at = models.DateTimeField(auto_now_add=True)




# models.py

from django.db import models
from django.contrib.auth.models import User

class UserToken(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)  # One-to-one relationship with User
    access_token = models.CharField(max_length=255)

    def __str__(self):
        return self.user.username




import csv
from django.db import models

class EventCSV(models.Model):
    file = models.FileField(upload_to='event_logs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def read_csv(self):
        data = []
        if self.file:
            self.file.open()
            reader = csv.DictReader(self.file)
            for row in reader:
                data.append(row)
            self.file.close()
        return data
