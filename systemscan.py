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

# Define folder paths
folder_path = "test/ransomware"
output_folder = "data"
predict_folder = "predict"
csv_filename = "system_scan.csv"
csv_path = os.path.join(output_folder, csv_filename)
# Check if folder_path is provided as command-line argument
if len(sys.argv) >= 2:
    folder_path = sys.argv[1]

def scan_and_write_csv(folder_path):
    print("Scanning files, calculating entropy and writing to csv")
    try:
        scanfile_csv(folder_path, csv_path)  # Call the scanfile_csv function with parameters
    except Exception as e:
        print(f"Error Scanning files, calculating entropy and writing to csv: {e}")

    print("Scan completed.")
    print(f"Copying system_scan.csv to {predict_folder}...")
    # Copy the system_scan.csv file to the predict folder
    shutil.copy(csv_path, os.path.join(predict_folder, csv_filename))
    print("Copy completed.")

    ransomware_predict()

def ransomware_predict():
    # Load data
    data = pd.read_csv("data/pretrained_data.csv")
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
    # Load the data from files_scan.csv
    new_data = pd.read_csv("predict/system_scan.csv")
    # Use the existing scaler instance fitted on the training data to scale the new data
    new_data_transformed = scaler.transform(new_data[['File Size', 'Entropy']])

    # Make predictions using Random Forest model
    predictions = ml.rf_predict(x, y, new_data_transformed)

    # Write the results back to the CSV file
    new_data['Ransomware'] = predictions
    new_data.to_csv("predict/system_predicted.csv", index=False)

# Schedule the job to run every 5 minutes
schedule.every(1).minutes.do(scan_and_write_csv, folder_path)

while True:
    schedule.run_pending()
    time.sleep(1)
