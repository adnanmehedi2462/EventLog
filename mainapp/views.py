

from django.shortcuts import render, get_object_or_404
import csv
import os
import joblib
import pandas as pd
import numpy as np
import chardet
from django.conf import settings
from rest_framework import generics
from .models import EventCSV
from .serializers import EventCSVSerializer
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from collections import Counter
from datetime import timedelta
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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



from datetime import timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

class TrainModelView(APIView):
    def get(self, request, pk):
        try:
            event_csv = EventCSV.objects.get(pk=pk)
            
            # Detect encoding
            with open(event_csv.file.path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Read the CSV with the detected encoding
            df = pd.read_csv(event_csv.file.path, encoding=encoding)

            df_filtered = df[df['EventType'].isin(['Error', 'Warning'])]
            df_filtered['TimeGenerated'] = pd.to_datetime(df_filtered['TimeGenerated'])

            # Define the date ranges
            today = pd.to_datetime("today")
            val_start_date = today - timedelta(days=80)  # 80 days ~ 2.5 months
            train_end_date = val_start_date - timedelta(days=1)
            train_start_date = train_end_date - timedelta(days=285)  # 285 days ~ 9.5 months

            # Train and validation splits
            df_train = df_filtered[(df_filtered['TimeGenerated'] >= train_start_date) & (df_filtered['TimeGenerated'] <= train_end_date)]
            df_val = df_filtered[df_filtered['TimeGenerated'] >= val_start_date]

            # Check for class imbalance
            print("EventType counts (training):", df_train['EventType'].value_counts())
            print("EventType counts (validation):", df_val['EventType'].value_counts())

            if 'EventType' in df_train.columns and not df_train.empty and not df_val.empty:
                # Prepare training data
                X_train = df_train.drop(columns=['EventType', 'TimeGenerated'], errors='ignore')
                X_train = pd.get_dummies(X_train)
                y_train = df_train['EventType']

                # Prepare validation data
                X_val = df_val.drop(columns=['EventType', 'TimeGenerated'], errors='ignore')
                X_val = pd.get_dummies(X_val)
                y_val = df_val['EventType']

                # Ensure X_val has the same columns as X_train
                X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

                # Handle class imbalance
                model = RandomForestClassifier(class_weight='balanced')

                # Train the model
                model.fit(X_train, y_train)

                # Evaluate the model
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average='weighted')
                print("Model accuracy:", accuracy)
                print("Model F1-score:", f1)

                # Save the model and metrics
                joblib.dump(model, f'models/model_{pk}.joblib')
                with open(f'models/model_{pk}_metrics.txt', 'w') as file:
                    file.write(f'accuracy: {accuracy}\nf1: {f1}')

                # Plot and save the timeline figures
                plt.figure(figsize=(12, 6))
                plt.plot(df_val['TimeGenerated'], y_val, label='Actual')
                plt.plot(df_val['TimeGenerated'], y_pred, label='Predicted', linestyle='--')
                plt.xlabel('Time')
                plt.ylabel('Event Occurrences')
                plt.title('Prediction vs Actual Data Timeline')
                plt.legend()
                plt.savefig(f'models/model_{pk}_timeline.png')

                return Response({"message": "Model trained successfully", "accuracy": accuracy, "f1": f1}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "'EventType' not found in DataFrame or no data to train"}, status=status.HTTP_400_BAD_REQUEST)

        except EventCSV.DoesNotExist:
            return Response({"error": "EventCSV not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)







import os
import chardet
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render, get_object_or_404
from .models import EventCSV
import json
from datetime import datetime, timedelta

def predict_maintenance(request, pk):
    event_csv = get_object_or_404(EventCSV, pk=pk)
    model_path = f'models/model_{pk}.joblib'
    
    if not os.path.exists(model_path):
        return render(request, 'prediction_report.html', {'error': 'Model not found'})

    model = joblib.load(model_path)
    
    # Load metrics
    metrics_path = f'models/model_{pk}_metrics.txt'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as file:
            metrics = file.read().splitlines()
            accuracy = float(metrics[0].split(': ')[1])
            f1 = float(metrics[1].split(': ')[1])
    else:
        accuracy = None
        f1 = None
    
    # Detect encoding
    with open(event_csv.file.path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    df = pd.read_csv(event_csv.file.path, encoding=encoding)

    # Filter for Error and Warning logs
    df_filtered = df[df['EventType'].isin(['Error', 'Warning'])]
    df_filtered['TimeGenerated'] = pd.to_datetime(df_filtered['TimeGenerated'])

    if df_filtered.empty:
        return render(request, 'prediction_report.html', {'error': 'No data available for prediction'})

    # Calculate training and testing periods
    today = datetime.now().date()
    validation_end = today
    validation_start = today - timedelta(days=80)  # 2.5 months
    training_end = validation_start - timedelta(days=1)
    training_start = training_end - timedelta(days=285)  # 9.5 months

    # Prepare features for prediction
    X = df_filtered.drop(columns=['EventType', 'TimeGenerated'], errors='ignore')
    X = pd.get_dummies(X)

    # Ensure X has the same columns as the training data
    model_columns = model.feature_names_in_
    X = X.reindex(columns=model_columns, fill_value=0)

    # Make predictions
    predictions = model.predict(X)
    prediction_probabilities = model.predict_proba(X)

    # Create a report
    report = df_filtered.copy()
    report['Predicted_EventType'] = predictions
    report['Predicted_Probability'] = np.max(prediction_probabilities, axis=1)

    # Manually set specific EventID probabilities
    manual_event_ids = [1, 100, 101, 102, 103]
    report.loc[report['EventID'].isin(manual_event_ids), 'Predicted_Probability'] = 0.01

    # Reset index
    report.reset_index(drop=True, inplace=True)

    # Calculate next occurrences based on average time interval
    next_occurrences = []
    today = pd.to_datetime("today")

    for event_id in report['EventID'].unique():
        occurrences = report[report['EventID'] == event_id]
        occurrences['TimeGenerated'] = pd.to_datetime(occurrences['TimeGenerated'])
        
        # Calculate time differences
        time_diffs = occurrences['TimeGenerated'].diff().dropna()
        
        # Calculate average time interval
        if not time_diffs.empty:
            avg_time_diff = time_diffs.mean()
            last_occurrence = occurrences['TimeGenerated'].max()
            next_occurrence = last_occurrence + avg_time_diff
            
            # Ensure next occurrence is in the future
            if next_occurrence <= today:
                next_occurrence = today + avg_time_diff
        else:
            next_occurrence = today  # Fallback to today

        next_occurrences.append(next_occurrence)

    # Map next occurrences to report
    report['Next_Occurrence'] = report['EventID'].map(dict(zip(report['EventID'].unique(), next_occurrences)))

    # Group by EventID and aggregate results
    report_grouped = report.groupby('EventID').agg(
        Date=('TimeGenerated', 'first'),
        EventType=('EventType', 'first'),
        Predicted_EventType=('Predicted_EventType', 'first'),
        Predicted_Probability=('Predicted_Probability', 'mean'),
        Next_Occurrence=('Next_Occurrence', 'first')
    ).reset_index()

    # Sort the report by Next_Occurrence date
    report_grouped = report_grouped.sort_values('Next_Occurrence')

    # Prepare chart data based on Next_Occurrence
    chart_data = {
        'labels': report_grouped['Next_Occurrence'].dt.strftime('%Y-%m-%d').tolist(),
        'datasets': [
            {
                'label': 'Error Probabilities',
                'data': [],
                'backgroundColor': 'rgba(255, 99, 132, 0.2)',
                'borderColor': 'rgba(255, 99, 132, 1)',
                'borderWidth': 1
            },
            {
                'label': 'Warning Probabilities',
                'data': [],
                'backgroundColor': 'rgba(54, 162, 235, 0.2)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'borderWidth': 1
            }
        ]
    }

    for _, row in report_grouped.iterrows():
        if row['EventType'] == 'Error':
            chart_data['datasets'][0]['data'].append(row['Predicted_Probability'] * 100)
            chart_data['datasets'][1]['data'].append(0)
        elif row['EventType'] == 'Warning':
            chart_data['datasets'][0]['data'].append(0)
            chart_data['datasets'][1]['data'].append(row['Predicted_Probability'] * 100)

    # Debug information
    print("Chart Data:", json.dumps(chart_data, indent=2))

    return render(request, 'prediction_report.html', {
        'report': report_grouped.to_dict(orient='records'),
        'accuracy': accuracy,
        'f1': f1,
        'chart_data': json.dumps(chart_data),
        'training_start': training_start.strftime('%Y-%m-%d'),
        'training_end': training_end.strftime('%Y-%m-%d'),
        'validation_start': validation_start.strftime('%Y-%m-%d'),
        'validation_end': validation_end.strftime('%Y-%m-%d'),
    })