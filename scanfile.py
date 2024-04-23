import os
import math
import hashlib
import csv
import shutil

def file_size_entropy_and_hash(file_path):
    """Calculate file size, Shannon entropy, and hash of a file."""
    try:
        # Calculate file size
        file_size = os.path.getsize(file_path)

        # Calculate Shannon entropy
        with open(file_path, 'rb') as f:
            byte_freq = {}
            total_bytes = 0
            for byte in f.read():
                byte_freq[byte] = byte_freq.get(byte, 0) + 1
                total_bytes += 1

            entropy = 0
            for count in byte_freq.values():
                probability = count / total_bytes
                entropy -= probability * math.log2(probability) 

        # Calculate file hash (MD5)
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        hash_value = hasher.hexdigest()

        return hash_value, file_size, entropy
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None, None
    
def scanfile_csv(folder_path, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Name', 'Hash', 'File Size', 'Entropy', 'Ransomware'])  # Write header

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                hash_value, file_size, entropy = file_size_entropy_and_hash(file_path)
                csv_writer.writerow([filename, hash_value, file_size, entropy, ''])  # Add a placeholder for ransomware column

# Define folder paths
folder_path = "test/ransomware"
output_folder = "data"
predict_folder = "predict"
csv_filename = "files_scan.csv"
csv_path = os.path.join(output_folder, csv_filename)

def scan_file_csv(folder_path):
    try:
        scanfile_csv(folder_path, csv_path)  # Call the scanfile_csv function with parameters
    except Exception as e:
        print(f"Error Scanning files, calculating entropy and writing to csv: {e}")
    # Copy the files_scan.csv file to the predict folder
    shutil.copy(csv_path, os.path.join(predict_folder, csv_filename))
    print("Scan & Copy completed.")
