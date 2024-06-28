# Generated by Django 5.0.6 on 2024-06-08 12:21

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='EventLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('time_generated', models.DateTimeField()),
                ('event_id', models.IntegerField()),
                ('event_type', models.IntegerField()),
                ('event_category', models.IntegerField()),
                ('source_name', models.CharField(max_length=255)),
                ('computer_name', models.CharField(max_length=255)),
                ('string_inserts', models.TextField(blank=True, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
