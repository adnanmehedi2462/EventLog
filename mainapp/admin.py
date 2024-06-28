# admin.py
from django.contrib import admin
from .models import EventLog

class EventTypeFilter(admin.SimpleListFilter):
    title = 'Event Type'
    parameter_name = 'event_type'

    def lookups(self, request, model_admin):
        return (
            ('1', 'Error (Critical)'),
            ('2', 'Warning'),
            ('4', 'Information (Normal)'),
        )

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(event_type=self.value())
        else:
            return queryset

@admin.register(EventLog)
class EventLogAdmin(admin.ModelAdmin):
    list_display = ('time_generated', 'source_name', 'event_id', 'event_type')
    list_filter = (EventTypeFilter,)

# admin.py

from django.contrib import admin
from .models import UserToken  # Import the correct model

class UserTokenAdmin(admin.ModelAdmin):
    list_display = ['user', 'access_token']

admin.site.register(UserToken, UserTokenAdmin)




from .models import EventCSV

# @admin.register(EventCSV)
# class EventCSVAdmin(admin.ModelAdmin):
#     list_display = ('id', 'file', 'uploaded_at')
#     list_filter = ('uploaded_at',)
#     search_fields = ('file',)
#     ordering = ('-uploaded_at',)
# mainapp/admin.py

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import EventCSV

class EventCSVAdmin(admin.ModelAdmin):
    list_display = ['file', 'uploaded_at', 'view_csv_link']

    def view_csv_link(self, obj):
        if obj.file:
            return format_html('<a href="{}" target="_blank">View CSV</a>', reverse('mainapp:admin_view_csv', args=[obj.id]))
        return '-'

    view_csv_link.allow_tags = True
    view_csv_link.short_description = 'View CSV'

admin.site.register(EventCSV, EventCSVAdmin)
