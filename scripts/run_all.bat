@echo off
REM Automated script for Breast Cancer Detection Project

REM Step 1: Install dependencies
pip install -r requirements.txt

REM Step 2: Train the model
python src\train\train.py

REM Step 3: Start the API server in a new window
start cmd /k python api\app.py

REM Step 4: Start the Streamlit web app in a new window
start cmd /k streamlit run webapp\streamlit_app.py

echo All steps initiated. You can now use the web app in your browser.
pause 