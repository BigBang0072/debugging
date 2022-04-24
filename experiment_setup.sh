#Creating the directories
mkdir nlp_logs
mkdir dataset 


#Downloading the multinli dataset
chmod +x multi_nli_downloader.sh
./multi_nli_downloader.sh

#Getting the python packages intalled
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg


#Setting up the git config
git config --global user.email "t-abkumar@microsoft.com"
git config --global user.name "Abhinav Kumar"
