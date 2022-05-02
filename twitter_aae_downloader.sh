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




#Downloading the PAN-16 dataset
#1. Get the request access on the website --> email will be sent with the link
#Add the token in the link we get to the downloader link to download on terminal
wget https://zenodo.org/record/3745963/files/pan16-author-profiling-training-dataset-2016-04-25.zip?token=eyJhbGciOiJIUzUxMiIsImV4cCI6MTY1NDEyMDc5OSwiaWF0IjoxNjUxNDc0NjY4fQ.eyJkYXRhIjp7InJlY2lkIjozNzQ1OTYzfSwiaWQiOjIyNzAyLCJybmQiOiJjNGY3YmNiNCJ9.MGnTbv5FqtPF-WeqmyBkgOZSkT_BVXqMr28twn_eyJXf_5x31_KBAaftaHJNS2GuUNom4-pJpeFNQLVdE-fmIw#.Ym-IWNpBxaQ