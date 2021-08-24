
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

# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 2 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 3 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 4 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 5 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 6 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 7 
# python nlp_models.py -expt_num "9.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 8 

# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 2 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 3 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 4 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 5 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 6 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 7 
# python nlp_models.py -expt_num "10.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 8 


#Successively increasing the number of example from different domain (but not a fair comparison)
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 2 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 3 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 4 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 5 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 6 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 7 
# python nlp_models.py -expt_num "11.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 8 


#Trying with pretrained fixed embedding. This should generalize and remove unwanted corr
#since, if somethin is bad in one, its nearby words will also become bad
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 2 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 3 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 4 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 5 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 6 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 7 
# python nlp_models.py -expt_num "12.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 8

#Training all the domain individually, for better performance in practice
#Otherwise ordering would have mattered.
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "arts"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "books"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "phones"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "clothes"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "groceries"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "movies"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "pets"
python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "tools"

#Increasing the dataset size per domain
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "arts"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "books"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "phones"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "clothes"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "groceries"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "movies"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "pets"
python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "tools"


#Using glove embedding
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "arts"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "books"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "phones"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "clothes"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "groceries"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "movies"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "pets"
python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "tools"

#Using the glove embedding + traininable
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "arts"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "books"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "phones"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "clothes"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "groceries"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "movies"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "pets"
python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "tools"

#Glove mebedding with larger dataset size
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "arts"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "books"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "phones"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "clothes"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "groceries"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "movies"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "pets"
python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "tools"

#Glove embedding + trainable + large dataet size
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "arts"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "books"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "phones"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "clothes"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "groceries"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "movies"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "pets"
python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "tools"
