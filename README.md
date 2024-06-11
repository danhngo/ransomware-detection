# ransomware-detection

# 1. Create Virttual Environmen
virtualenv venv
python3 -m venv venv
source venv/bin/activate

# 2. Install requirment
pip3 install -r requirements.txt

# 3. Run App
python3 app.py

# 4. Extract file attributes and train model
python3 feature_extraction.py

Note: the training files are placed in "training" folder, same level with project folder.
    - ransomware-detection
    - training  
    - test

# 5. Cronjob to scan & detect ransomware at a specific folder
python3 ransomware-detection.py

Note: the test files are placed in "test" folder, same level with project folder.
    - ransomware-detection
    - test
    - training  

# 6. Run at server
nohup python3 app.py &
Server running at: http://172.208.119.43:8060

# 7. Server Info
ssh danh@172.208.119.43
git clone git@github.com:danhngo/ransomware-detection.git
/home/danh/ransomware-detection
lsof -i tcp:8060





 
 