# ransomware-detection

# 1. Create Virttual Environmen
virtualenv venv
python3 -m venv venv
source venv/bin/activate

# 2. Install requirment
pip3 install -r requirements.txt

# 3. Run App
python3 app.py

# 4. Run extract file attributes and save csv
python3 feature_extraction.py

# 5. Cronjob to scan & detect ransomware at a specific folder
python3 ransomware-detection.py

# 6. Run at server
nohup python3 app.py &

# 7. Server Info
ssh danh@172.208.119.43
git clone git@github.com:danhngo/ransomware-detection.git
cd /home/danh/ransomware-detection-main
http://172.208.119.43:8060/

lsof -i tcp:8060

pip3 install opencv-python

 
 