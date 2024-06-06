 # Base file to scan files in a folder, calculate entropy, and write to a CSV file
import os
import math
import hashlib
import csv
import shutil

import os
import math
import hashlib

def file_size_entropy_and_hash(file_path, header_size=512):
    """Calculate file size, Shannon entropy, hash, and header size of a file."""
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

        # Calculate header size
        with open(file_path, 'rb') as f:
            file_header = f.read(header_size)
        actual_header_size = len(file_header)

        return hash_value, file_size, entropy, actual_header_size
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None, None, None


