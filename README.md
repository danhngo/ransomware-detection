# ransomware-detection

# 1. Create Virttual Environmen
virtualenv venv
python3 -m venv venv
source venv/bin/activate

# 2. Install requirment
pip3 install -r requirement.txt

# 3. Run App
python3 app.py

# 4. Run extract file attributes and save csv
python3 feature_extraction.py

# 5. Cronjob to scan & detect ransomware at a specific folder
python3 ransomware-detection.py

# 6. Run at server
nohup python3 app.py &

# 7. Server Info
ssh docker@40.121.214.159 
cd /home/docker/ransomware-detection-main
http://40.121.214.159:8050/

lsof -i tcp:8050
https://coolors.co/palettes/trending

pip3 install opencv-python

 
 