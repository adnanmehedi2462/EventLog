# Generated by Django 5.0.6 on 2024-06-08 14:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mainapp', '0003_alter_eventlog_computer_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='eventlog',
            name='computer_name',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='eventlog',
            name='event_category',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='eventlog',
            name='event_id',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='eventlog',
            name='event_type',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='eventlog',
            name='source_name',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='eventlog',
            name='time_generated',
            field=models.TextField(blank=True, null=True),
        ),
    ]
