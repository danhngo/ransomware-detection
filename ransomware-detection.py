import os
import csv
import scanfile as scan
from datetime import datetime
import schedule
import time
import pandas as pd
import ml
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

folder_paths = "/Users/admin/11.SaskPoly/4.capstone/test/files"  # Add more folder paths as needed
train_folder = "data"
pretrain_data = "pretrained_data.csv"
output_file = "scan_files.csv"
predict_folder = "predict"


def scan_all(folder_path):
    print(f"Scan Folder and subfolders of {folder_path}")
    try:
        file_details = scan_all_files(folder_path)

        # Get the current directory
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        # Define folder paths
        print(f"Parent is {parent_directory}")
        output_folder = os.path.join(parent_directory, "scan")

        csv_filename = os.path.join(output_folder, output_file)
        #csv_filename = f"{os.path.basename(folder_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        write_to_csv(file_details, csv_filename)
        print("Write files completed.")
        ransomware_detection()
        print("Ransomware detection completed.")
    except Exception as e:
        print(f"Error scanning files and writing to CSV: {e}")

def scan_all_files(folder_path):
    """Recursively scan all files in the folder and subfolders."""
    file_details = []
    today_date = datetime.today().strftime('%Y-%m-%d')
    for root, dirs, files in os.walk(folder_path):
        print(f"Start scanning folder {root}...")
        for file_name in files:
            if file_name != ".DS_Store":
                file_path = os.path.join(root, file_name)
                hash_value, file_size, entropy = scan.file_size_entropy_and_hash(file_path)
                # Determine if entropy is high
                #high_entropy = 1 if entropy is not None and entropy > 7.94 else 0
                _, file_extension = os.path.splitext(file_name)
                file_details.append((file_name, hash_value,file_extension, file_size, entropy, today_date,''))
                    
    return file_details


def write_to_csv(file_details, csv_filename):
    """Write file details to a CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Name', 'Hash', 'File Type', 'File Size', 'Entropy','Date','Ransomware'])  # Write header
        csv_writer.writerows(file_details)

def ransomware_detection():
    # Load data
    data = pd.read_csv(os.path.join(train_folder, pretrain_data))
    # Standardize data
    scaler = StandardScaler()
    data[['File Size', 'Entropy']] = scaler.fit_transform(data[['File Size', 'Entropy']])
    # Load data
    x = data[['File Size', 'Entropy']]
    y = data['Ransomware']
    rf_accuracy, _, _, _ = ml.rf_evaluate(x, y)
    print(f"Model evaluated, accuracy: {rf_accuracy}")
    print("Starting ransomware prediction...")
  
    # Load the data from scan_files.csv
    # Get the current directory
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    # Define folder paths
    print(f"Parent is {parent_directory}")
    output_folder = os.path.join(parent_directory, "scan")
    print(f"Read data from {output_folder}")
    new_data = pd.read_csv(os.path.join(output_folder, output_file))
    # Use the existing scaler instance fitted on the training data to scale the new data
    new_data_transformed = scaler.transform(new_data[['File Size', 'Entropy']])
    # Make predictions using Random Forest model
    predictions = ml.rf_predict(x, y, new_data_transformed)

    # Write the results back to the CSV file
    new_data['Ransomware'] = predictions
    new_data.to_csv(os.path.join(predict_folder, "ransomware_detection.csv"), index=False)
           

# Schedule the job to run every 5 minutes
schedule.every(1).minutes.do(scan_all, folder_paths)

while True:
    schedule.run_pending()
    time.sleep(1)

