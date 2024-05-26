# ransomware-detection

# 1. Create Virttual Environmen
virtualenv venv
python3 -m venv venv
source venv/bin/activate

# 2. Install requirment
pip3 install -r requirement.txt

# 3. Run App
python3 app.py
python3 cronjob.py

# 4. Run App
ssh docker@40.121.214.159 

cd /home/docker/ransomware-detection-main

nohup python3 app.py &
http://40.121.214.159:8050/
lsof -i tcp:8050
https://coolors.co/palettes/trending


pip3 install opencv-python

 
 