#Script inspired from:
#https://github.com/yanaiela/demog-text-removal/tree/master/src/data

#Now downloading the preprocessor repository
git clone git@github.com:yanaiela/demog-text-removal.git
conda create -n twitteraae python=2.7
conda activate twitteraae
pip install -r demog-text-removal/requirements.txt

#Getting the dataset files
cd ./demog-text-removal/src/data/
wget http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip
unzip TwitterAAE-full-v1.zip

#Running the data conversion process
mkdir sentiment_race
python make_data.py ./TwitterAAE-full-v1/twitteraae_all ./sentiment_race sentiment race


