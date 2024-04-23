import os
import csv
import scanfile as scan

def scan_all_files(folder_path):
    """Recursively scan all files in the folder and subfolders."""
    file_details = []
    for root, dirs, files in os.walk(folder_path):
        print(f"Start scanning folder {root}...")
        for file_name in files:
            if file_name != ".DS_Store":
                file_path = os.path.join(root, file_name)
                hash_value, file_size, entropy = scan.file_size_entropy_and_hash(file_path)
                # Determine ransomware value based on folder or subfolder name
                ransomware = 1 if "ransomware" in root.lower() else 0
                # Determine if entropy is high
                high_entropy = 1 if entropy is not None and entropy > 7.94 else 0
                if ransomware == 1 or (ransomware == 0 and (entropy is None or entropy < 7.95)):
                    # Extract file extension
                    _, file_extension = os.path.splitext(file_name)
                    # Append file details to the list
                    file_details.append((file_name, hash_value, file_extension, file_size, entropy, ransomware,high_entropy))
    return file_details


def write_to_csv(file_details, csv_filename):
    """Write file details to a CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Name', 'Hash', 'File Type', 'File Size', 'Entropy', 'Ransomware','High Entropy'])  # Write header
        csv_writer.writerows(file_details)

def scan_all(folder_path, csv_filename):
    """Scan all files in the folder and subfolders, and write details to a CSV file."""
    print("Scanning files, calculating entropy, and writing to CSV...")
    try:
        file_details = scan_all_files(folder_path)
        write_to_csv(file_details, csv_filename)
        print("Write files completed.")
    except Exception as e:
        print(f"Error scanning files and writing to CSV: {e}")

if __name__ == "__main__":
    # Get the current directory
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    # Define folder paths
    print(f"Parent {parent_directory}...")
    folder_path = os.path.join(parent_directory, "training")
    print(f"Folder {folder_path}...")
    output_folder = "data"
    csv_filename = "pretrained_data.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    # Scan files and write to CSV
    scan_all(folder_path, csv_path)

# Define folder paths
# folder_path = "test/ransomware"
# output_folder = "data"
# predict_folder = "predict"
# csv_filename = "files_scan_all.csv"
# csv_path = os.path.join(output_folder, csv_filename)
# # Check if folder_path is provided as command-line argument
# if len(sys.argv) >= 2:
#     folder_path = sys.argv[1]

# def scan_and_write_csv(folder_path):
#     print("Scanning files, calculating entropy and writing to csv")
#     try:
#         scan.scanfile_csv(folder_path, csv_path)  # Call the scanfile_csv function with parameters
#     except Exception as e:
#         print(f"Error Scanning files, calculating entropy and writing to csv: {e}")

#     print("Scan completed.")
#     print(f"Copying files_scan.csv to {predict_folder}...")
#     # Copy the files_scan.csv file to the predict folder
#     shutil.copy(csv_path, os.path.join(predict_folder, csv_filename))
#     print("Copy completed.")

# Schedule the job to run every 5 minutes
#schedule.every(5).minutes.do(scan_all, folder_path,csv_filename)

#while True:
#    schedule.run_pending()
#    time.sleep(1)
