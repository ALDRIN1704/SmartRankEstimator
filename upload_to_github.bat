@echo off
cd /d B:\subjects\sem6\GITHUB\projectPredictRank

REM Initialize Git
git init

REM Create .gitignore
echo __pycache__/ > .gitignore
echo *.pyc >> .gitignore
echo *.pkl >> .gitignore
echo .env >> .gitignore
echo .DS_Store >> .gitignore

REM Create requirements.txt
echo pandas==1.5.3 > requirements.txt
echo numpy==1.24.1 >> requirements.txt
echo scikit-learn==1.2.1 >> requirements.txt
echo flask==2.2.3 >> requirements.txt
echo joblib==1.2.0 >> requirements.txt

REM Commit and Push
git add .
git commit -m "Initial commit with all project files"
git remote add origin https://github.com/ALDRIN1704/rankgenie-ml-rank-predictor.git
git branch -M main
git push -u origin main

pause
