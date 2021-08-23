
# #Large IMDB movie training
# python nlp_models.py -expt_num "5.fd1.single" -emb_path "random" -emb_train True -normalize_emb True
# python nlp_models.py -expt_num "5.fd1.both" -emb_path "random" -emb_train True -normalize_emb True

# python nlp_models.py -expt_num "6.fd1.single" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False
# python nlp_models.py -expt_num "6.fd1.both" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False

# python nlp_models.py -expt_num "7.fd1.single" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False
# python nlp_models.py -expt_num "7.fd1.both" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False

# python nlp_models.py -expt_num "8.fd1.single" -emb_path "random" -emb_train True -normalize_emb False
# python nlp_models.py -expt_num "8.fd1.both" -emb_path "random" -emb_train True -normalize_emb False

mkdir logs

python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 7000 -num_domains 2 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 6000 -num_domains 3 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 5000 -num_domains 4 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 4000 -num_domains 5 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 3000 -num_domains 6 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 2000 -num_domains 7 
python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 1000 -num_domains 8 