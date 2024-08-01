

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
            print("Filtered EventIDs:", df_filtered['EventID'].unique())

            if 'EventType' in df_filtered.columns and not df_filtered.empty:
                X = df_filtered.drop(columns=['EventType'], errors='ignore')
                X = pd.get_dummies(X)
                y = df_filtered['EventType']

                # Check for class imbalance
                print("EventType counts:", y.value_counts())

                # Optional: Handle class imbalance
                model = RandomForestClassifier(class_weight='balanced')

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train the model
                model.fit(X_train, y_train)

                # Evaluate the model
                accuracy = model.score(X_test, y_test)
                print("Model accuracy:", accuracy)

                # Save the model and accuracy
                joblib.dump(model, f'models/model_{pk}.joblib')
                with open(f'models/model_{pk}_accuracy.txt', 'w') as file:
                    file.write(str(accuracy))

                return Response({"message": "Model trained successfully", "accuracy": accuracy}, status=status.HTTP_200_OK)
            else:
                return Response({"error": "'EventType' not found in DataFrame or no data to train"}, status=status.HTTP_400_BAD_REQUEST)

        except EventCSV.DoesNotExist:
            return Response({"error": "EventCSV not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





def predict_maintenance(request, pk):
    event_csv = get_object_or_404(EventCSV, pk=pk)
    model_path = f'models/model_{pk}.joblib'
    
    if not os.path.exists(model_path):
        return render(request, 'prediction_report.html', {'error': 'Model not found'})

    model = joblib.load(model_path)
    
    # Load accuracy
    accuracy_path = f'models/model_{pk}_accuracy.txt'
    if os.path.exists(accuracy_path):
        with open(accuracy_path, 'r') as file:
            accuracy = float(file.read())
    else:
        accuracy = None
    
    # Detect encoding
    with open(event_csv.file.path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    df = pd.read_csv(event_csv.file.path, encoding=encoding)

    # Filter for Error and Warning logs
    df_filtered = df[df['EventType'].isin(['Error', 'Warning'])]

    if df_filtered.empty:
        return render(request, 'prediction_report.html', {'error': 'No data available for prediction'})

    # Prepare features for prediction
    X = df_filtered.drop(columns=['EventType'], errors='ignore')
    X = pd.get_dummies(X)

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

    return render(request, 'prediction_report.html', {'report': report_grouped.to_dict(orient='records'), 'accuracy': accuracy})
