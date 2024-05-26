# cron job to scan files, calculate entropy, and write to CSV
import schedule
import time
import sys
import os
import shutil
from scanfile import scanfile_csv
import scanfile as scan
import pandas as pd
import ml
from sklearn.preprocessing import StandardScaler

# Suppress warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

# Define folder paths
folder_paths = ["/Users/admin/Downloads", "test/files"]  # Add more folder paths as needed
output_folder = "data"
predict_folder = "predict"
csv_filename = "system_scan.csv"

def scan_and_write_csv(folder_paths):
    print("Scanning files, calculating entropy and writing to csv")
    try:
        for folder_path in folder_paths:
            scanfile_csv(folder_path, os.path.join(output_folder, csv_filename))  # Call the scanfile_csv function for each folder path
    except Exception as e:
        print(f"Error scanning files, calculating entropy, and writing to csv: {e}")

    print("Scan completed.")
    print(f"Copying {csv_filename} to {predict_folder}...")
    # Copy the system_scan.csv file to the predict folder for each folder path
    for folder_path in folder_paths:
        shutil.copy(os.path.join(output_folder, csv_filename), os.path.join(predict_folder, f"{os.path.basename(folder_path)}_{csv_filename}"))
    print("Copy completed.")

    ransomware_predict()

def ransomware_predict():
    # Load data
    data = pd.read_csv(os.path.join(output_folder, "pretrained_data.csv"))
    # Standardize data
    scaler = StandardScaler()
    data[['File Size', 'Entropy']] = scaler.fit_transform(data[['File Size', 'Entropy']])
    # Load data
    x = data[['File Size', 'Entropy']]
    y = data['Ransomware']
    rf_accuracy, _, _, _ = ml.rf_evaluate(x, y)
    print(f"Model evaluated, accuracy: {rf_accuracy}")
    print("Starting ransomware prediction...")
    # Calculate file size, entropy, and write to CSV
    for folder_path in folder_paths:
        # Load the data from system_scan.csv
        new_data = pd.read_csv(os.path.join(output_folder, csv_filename))

        # Use the existing scaler instance fitted on the training data to scale the new data
        new_data_transformed = scaler.transform(new_data[['File Size', 'Entropy']])

        # Make predictions using Random Forest model
        predictions = ml.rf_predict(x, y, new_data_transformed)

        # Write the results back to the CSV file
        new_data['Ransomware'] = predictions
        new_data.to_csv(os.path.join(predict_folder, f"{os.path.basename(folder_path)}_predicted.csv"), index=False)

# Schedule the job to run every 5 minutes
schedule.every(1).minutes.do(scan_and_write_csv, folder_paths)

while True:
    schedule.run_pending()
    time.sleep(1)
