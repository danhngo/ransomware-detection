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

#4. Run on server

 nohup python3 app.py &
 http://40.121.214.159:8050/

 lsof -i tcp:8050
 