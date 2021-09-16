
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
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "arts"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "books"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "phones"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "clothes"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "groceries"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "movies"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "pets"
# python nlp_models.py -expt_num "13.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 8000 -num_domains 1 -domain_name "tools"

# #Increasing the dataset size per domain
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "arts"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "books"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "phones"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "clothes"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "groceries"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "movies"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "pets"
# python nlp_models.py -expt_num "14.amzn" -emb_path "random" -emb_train True -normalize_emb True -num_samples 16000 -num_domains 1 -domain_name "tools"


# #Using glove embedding
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "arts"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "books"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "phones"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "clothes"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "groceries"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "movies"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "pets"
# python nlp_models.py -expt_num "15.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "tools"

# #Using the glove embedding + traininable
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "arts"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "books"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "phones"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "clothes"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "groceries"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "movies"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "pets"
# python nlp_models.py -expt_num "16.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 8000 -num_domains 1 -domain_name "tools"

# #Glove mebedding with larger dataset size
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "arts"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "books"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "phones"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "clothes"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "groceries"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "movies"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "pets"
# python nlp_models.py -expt_num "17.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train False -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "tools"

# #Glove embedding + trainable + large dataet size
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "arts"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "books"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "phones"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "clothes"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "groceries"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "movies"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "pets"
# python nlp_models.py -expt_num "18.amzn" -emb_path "glove-wiki-gigaword-100" -emb_train True -normalize_emb False -num_samples 16000 -num_domains 1 -domain_name "tools"


mkdir nlp_logs

#Increasing the number of topics
# python transformer_debugger.py -expt_num "2.0" -num_samples 100 -num_topics 10 -tfreq_ulim 0.7 -train_bert False
# python transformer_debugger.py -expt_num "2.1" -num_samples 100 -num_topics 20 -tfreq_ulim 0.7 -train_bert False
# python transformer_debugger.py -expt_num "2.2" -num_samples 100 -num_topics 40 -tfreq_ulim 0.7 -train_bert False

# #Increasing the frequency cutoof
# python transformer_debugger.py -expt_num "6.0" -num_samples 100 -num_topics 10 -tfreq_ulim 0.95 -train_bert False
# python transformer_debugger.py -expt_num "6.1" -num_samples 100 -num_topics 20 -tfreq_ulim 0.95 -train_bert False
# python transformer_debugger.py -expt_num "6.2" -num_samples 100 -num_topics 40 -tfreq_ulim 0.95 -train_bert False

# #Decreasing the frequency cutoof
# python transformer_debugger.py -expt_num "3.0" -num_samples 100 -num_topics 10 -tfreq_ulim 0.5 -train_bert False
# python transformer_debugger.py -expt_num "3.1" -num_samples 100 -num_topics 20 -tfreq_ulim 0.5 -train_bert False
# python transformer_debugger.py -expt_num "3.2" -num_samples 100 -num_topics 40 -tfreq_ulim 0.5 -train_bert False

# #Increasing the number of sample
# python transformer_debugger.py -expt_num "4.0" -num_samples 500 -num_topics 10 -tfreq_ulim 0.7 -train_bert False
# python transformer_debugger.py -expt_num "4.1" -num_samples 500 -num_topics 20 -tfreq_ulim 0.7 -train_bert False
# python transformer_debugger.py -expt_num "4.2" -num_samples 500 -num_topics 40 -tfreq_ulim 0.7 -train_bert False

# #Now training the bert
# python transformer_debugger.py -expt_num "5.0" -num_samples 100 -num_topics 10 -tfreq_ulim 0.7 -train_bert True 
# python transformer_debugger.py -expt_num "5.1" -num_samples 100 -num_topics 20 -tfreq_ulim 0.7 -train_bert True 
# python transformer_debugger.py -expt_num "5.2" -num_samples 100 -num_topics 40 -tfreq_ulim 0.7 -train_bert True

#Testing the manually creafted topics
# python transformer_debugger.py -expt_num "7.0" -num_samples 100 -num_topics 10 -num_epochs 5                         -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.1" -num_samples 100 -num_topics 10 -num_epochs 5  -load_weight "7.0"     -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.2" -num_samples 500 -num_topics 10 -num_epochs 5                         -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.3" -num_samples 500 -num_topics 10 -num_epochs 5  -load_weight "7.2"     -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.4" -num_samples 2500 -num_topics 10 -num_epochs 10                         -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.5" -num_samples 5000 -num_topics 10 -num_epochs 1                         -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.6" -num_samples 2500 -num_topics 10 -num_epochs 10                         -transformer "bert-base-uncased"


#using distilbert
# python transformer_debugger.py -expt_num "8.0" -num_samples 100 -num_topics 10 -num_epochs 5                         -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "8.1" -num_samples 100 -num_topics 10 -num_epochs 5  -load_weight "8.0"     -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "8.2" -num_samples 500 -num_topics 10 -num_epochs 5                         -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "8.3" -num_samples 500 -num_topics 10 -num_epochs 5  -load_weight "8.2"     -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "8.4" -num_samples 2500 -num_topics 10 -num_epochs 5                         -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "8.5" -num_samples 2500 -num_topics 10 -num_epochs 5  -load_weight "8.4"     -transformer "distilbert-base-uncased"



# python transformer_debugger.py -expt_num "7.1" -num_samples 100 -num_topics 10 -train_bert False -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "7.2" -num_samples 200 -num_topics 10 -train_bert False -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.3" -num_samples 200 -num_topics 10 -train_bert False -transformer "distilbert-base-uncased"
# python transformer_debugger.py -expt_num "7.4" -num_samples 100 -num_topics 10 -train_bert True -transformer "bert-base-uncased"
# python transformer_debugger.py -expt_num "7.5" -num_samples 100 -num_topics 10 -train_bert True -transformer "distilbert-base-uncased"







#Testing the effect of increating topic dataset size on the spuriousness topics
# python transformer_debugger.py -expt_num "8.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" # 0.6k/topic *2
# python transformer_debugger.py -expt_num "9.0" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" # 1.2k/topic *2
# python transformer_debugger.py -expt_num "9.1" -num_samples 4000 -num_topic_samples 2400 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" # 2.4k/topic *2
# python transformer_debugger.py -expt_num "9.2" -num_samples 8000 -num_topic_samples 4800 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" # 4.8k/topic *2


#Testing the gate effect 
# python transformer_debugger.py -expt_num "8.1.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "8.1" -gate_weight_epoch 2 -gate_var_cutoff 0.75
# python transformer_debugger.py -expt_num "8.1.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "8.1" -gate_weight_epoch 2 -gate_var_cutoff 0.50

# python transformer_debugger.py -expt_num "9.0.1" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "9.0" -gate_weight_epoch 2 -gate_var_cutoff 0.75
# python transformer_debugger.py -expt_num "9.0.2" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "9.0" -gate_weight_epoch 2 -gate_var_cutoff 0.50

# python transformer_debugger.py -expt_num "9.1.1" -num_samples 4000 -num_topic_samples 2400 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "9.1" -gate_weight_epoch 2 -gate_var_cutoff 0.75
# python transformer_debugger.py -expt_num "9.1.2" -num_samples 4000 -num_topic_samples 2400 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "9.1" -gate_weight_epoch 2 -gate_var_cutoff 0.50

# python transformer_debugger.py -expt_num "9.2.1" -num_samples 8000 -num_topic_samples 4800 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "9.2" -gate_weight_epoch 2 -gate_var_cutoff 0.75
# python transformer_debugger.py -expt_num "9.2.2" -num_samples 8000 -num_topic_samples 4800 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -gate_weight_exp "9.2" -gate_weight_epoch 2 -gate_var_cutoff 0.25


#Synthetic experiments (sample variantion)
# python transformer_debugger.py -expt_num "syn.5.0" -num_samples 500 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.5.1" -num_samples 1000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.5.2" -num_samples 2000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.5.3" -num_samples 4000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.5.4" -num_samples 8000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1

#(topic variation)
# python transformer_debugger.py -expt_num "syn.11.0" -num_samples 1000 -num_epochs 10 -num_cat 2 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.1" -num_samples 1000 -num_epochs 10 -num_cat 4 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.2" -num_samples 1000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.3" -num_samples 1000 -num_epochs 10 -num_cat 16 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.4" -num_samples 1000 -num_epochs 10 -num_cat 32 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.5" -num_samples 1000 -num_epochs 10 -num_cat 64 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.6" -num_samples 1000 -num_epochs 10 -num_cat 128 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1
# python transformer_debugger.py -expt_num "syn.11.7" -num_samples 1000 -num_epochs 10 -num_cat 256 -num_causal_nodes 1 -num_child_nodes 8 -num_topics 1

# python transformer_debugger.py -expt_num "syn.12.0" -num_samples 1000 -num_epochs 10 -num_cat 1 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.1" -num_samples 1000 -num_epochs 10 -num_cat 2 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.2" -num_samples 1000 -num_epochs 10 -num_cat 3 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.3" -num_samples 1000 -num_epochs 10 -num_cat 4 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.3" -num_samples 1000 -num_epochs 10 -num_cat 5 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.4" -num_samples 1000 -num_epochs 10 -num_cat 6 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.5" -num_samples 1000 -num_epochs 10 -num_cat 7 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1
# python transformer_debugger.py -expt_num "syn.12.6" -num_samples 1000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 2 -num_topics 1


# python transformer_debugger.py -expt_num "syn.13.0" -num_samples 1000 -num_epochs 10 -num_cat 2 -num_causal_nodes 1 -num_child_nodes 5 -num_topics 1
# python transformer_debugger.py -expt_num "syn.13.1" -num_samples 1000 -num_epochs 10 -num_cat 4 -num_causal_nodes 1 -num_child_nodes 5 -num_topics 1
# python transformer_debugger.py -expt_num "syn.13.2" -num_samples 1000 -num_epochs 10 -num_cat 8 -num_causal_nodes 1 -num_child_nodes 5 -num_topics 1
# python transformer_debugger.py -expt_num "syn.13.3" -num_samples 1000 -num_epochs 10 -num_cat 16 -num_causal_nodes 1 -num_child_nodes 5 -num_topics 1
# python transformer_debugger.py -expt_num "syn.13.4" -num_samples 1000 -num_epochs 10 -num_cat 32 -num_causal_nodes 1 -num_child_nodes 5 -num_topics 1


python transformer_debugger.py -expt_num "syn.16.0" -num_samples 1000 -num_epochs 30 -num_cat 3 -num_causal_nodes 1 -num_child_nodes 3 -num_topics 1
python transformer_debugger.py -expt_num "syn.16.1" -num_samples 1000 -num_epochs 30 -num_cat 5 -num_causal_nodes 1 -num_child_nodes 3 -num_topics 1
python transformer_debugger.py -expt_num "syn.16.2" -num_samples 1000 -num_epochs 30 -num_cat 7 -num_causal_nodes 1 -num_child_nodes 3 -num_topics 1
python transformer_debugger.py -expt_num "syn.16.3" -num_samples 1000 -num_epochs 30 -num_cat 9 -num_causal_nodes 1 -num_child_nodes 3 -num_topics 1
python transformer_debugger.py -expt_num "syn.16.4" -num_samples 1000 -num_epochs 30 -num_cat 11 -num_causal_nodes 1 -num_child_nodes 3 -num_topics 1


