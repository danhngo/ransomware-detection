import os
import csv
import scanfile as scan
from datetime import datetime

def scan_all(folder_path, csv_filename):
    print("Scanning files, calculating entropy, and writing to CSV...")
    try:
        file_details = scan_all_files(folder_path)
        write_to_csv(file_details, csv_filename)
        print("Write files completed.")
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
                hash_value, file_size, entropy,header_size = scan.file_size_entropy_and_hash(file_path)
                # Determine ransomware value based on folder or subfolder name
                ransomware = 1 if "ransomware" in root.lower() else 0
                # Determine ransomware type based on folder or subfolder name
                ransomware_type = ""
                if "dharma" in root.lower():
                    ransomware_type = "DHARMA"
                elif "maze" in root.lower():
                    ransomware_type = "MAZE"
                elif "netwalker" in root.lower():
                    ransomware_type = "NETWALKER"
                elif "phobos" in root.lower():
                    ransomware_type = "PHOBOS"
                elif "sodinokibi" in root.lower():
                    ransomware_type = "SODINOKIBI"
                elif "lockbit" in root.lower():
                    ransomware_type = "LOCKBIT"
                elif "ryuk" in root.lower():
                    ransomware_type = "RYUK"
                elif "conti" in root.lower():
                    ransomware_type = "CONTI"
                elif "revil" in root.lower():
                    ransomware_type = "REVIL"
                elif "wannacry" in root.lower():
                    ransomware_type = "WANNACRY"
                elif "petya" in root.lower():
                    ransomware_type = "PETYA"
                elif "cerber" in root.lower():
                    ransomware_type = "CERBER"
                elif "cryptolocker" in root.lower():
                    ransomware_type = "CRYPTOLOCKER"
                # Determine if entropy is high
                high_entropy = 1 if entropy is not None and entropy > 7.94 else 0
                if ransomware == 1 or (ransomware == 0 and (entropy is None or entropy < 7.95)):
                    # Extract file extension
                    _, file_extension = os.path.splitext(file_name)
                    # Append file details to the list
                    file_details.append((file_name, hash_value, file_extension, file_size, entropy,header_size, ransomware,ransomware_type,high_entropy,today_date))
    return file_details


def write_to_csv(file_details, csv_filename):
    """Write file details to a CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File Name', 'Hash', 'File Type', 'File Size', 'Entropy','Header Size', 'Ransomware','Ransomware Type','High Entropy','Date'])  # Write header
        csv_writer.writerows(file_details)

if __name__ == "__main__":
    # Get the current directory
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    # Define folder paths
    print(f"Parent is {parent_directory}")
    folder_path = os.path.join(parent_directory, "training")
    print(f"Scan Folder and subfolders of {folder_path}")
    csv_path = os.path.join("data", "pretrained_data.csv")
    # Scan files and write to CSV
    scan_all(folder_path, csv_path)
