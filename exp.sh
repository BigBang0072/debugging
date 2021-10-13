
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


#Testing the L1 loss
# python transformer_debugger.py -expt_num "10.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0" -load_weight_epoch 2 # 0.6k/topic *2
# python transformer_debugger.py -expt_num "10.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1" -load_weight_epoch 2
# python transformer_debugger.py -expt_num "10.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2" -load_weight_epoch 2
# python transformer_debugger.py -expt_num "10.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3" -load_weight_epoch 2
# python transformer_debugger.py -expt_num "10.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 100.0 -load_weight_exp "10.4" -load_weight_epoch 2

#Gating the previous experiment with new spuriousness score based on perf drop
# python transformer_debugger.py -expt_num "10.0.gate.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -gate_weight_exp "10.0" -gate_weight_epoch 2 -gate_var_cutoff "neg"
# python transformer_debugger.py -expt_num "10.0.gate.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -gate_weight_exp "10.0" -gate_weight_epoch 2 -gate_var_cutoff "0.1"
# python transformer_debugger.py -expt_num "10.0.gate.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0.gate.2" -load_weight_epoch 2


# python transformer_debugger.py -expt_num "10.0.gate.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -gate_weight_exp "10.0" -gate_weight_epoch 2 -gate_var_cutoff "0.4"
# python transformer_debugger.py -expt_num "10.0.gate.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0.gate.3" -load_weight_epoch 2

# python transformer_debugger.py -expt_num "10.0.gate.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0.gate.0" -load_weight_epoch 2 -gate_var_cutoff "0.05"
# python transformer_debugger.py -expt_num "10.0.gate.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0.gate.1" -load_weight_epoch 2 -gate_var_cutoff "0.2"
# python transformer_debugger.py -expt_num "10.0.gate.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0.gate.4" -load_weight_epoch 2 -gate_var_cutoff "0.6"
# python transformer_debugger.py -expt_num "10.0.gate.5" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 -load_weight_exp "10.0.gate.5" -load_weight_epoch 2 -gate_var_cutoff "0.8"




# python transformer_debugger.py -expt_num "10.1.gate.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1.gate.0" -load_weight_epoch 2 -gate_var_cutoff "0.05"
# python transformer_debugger.py -expt_num "10.1.gate.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1.gate.1" -load_weight_epoch 2 -gate_var_cutoff "0.1"
# python transformer_debugger.py -expt_num "10.1.gate.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1.gate.2" -load_weight_epoch 2 -gate_var_cutoff "0.2"
# python transformer_debugger.py -expt_num "10.1.gate.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1.gate.3" -load_weight_epoch 2 -gate_var_cutoff "0.4"
# python transformer_debugger.py -expt_num "10.1.gate.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1.gate.4" -load_weight_epoch 2 -gate_var_cutoff "0.6"
# python transformer_debugger.py -expt_num "10.1.gate.5" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0 -load_weight_exp "10.1.gate.5" -load_weight_epoch 2 -gate_var_cutoff "0.8"


# python transformer_debugger.py -expt_num "10.2.gate.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2.gate.0" -load_weight_epoch 2 -gate_var_cutoff "0.05"
# python transformer_debugger.py -expt_num "10.2.gate.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2.gate.1" -load_weight_epoch 2 -gate_var_cutoff "0.1"
# python transformer_debugger.py -expt_num "10.2.gate.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2.gate.2" -load_weight_epoch 2 -gate_var_cutoff "0.2"
# python transformer_debugger.py -expt_num "10.2.gate.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2.gate.3" -load_weight_epoch 2 -gate_var_cutoff "0.4"
# python transformer_debugger.py -expt_num "10.2.gate.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2.gate.4" -load_weight_epoch 2 -gate_var_cutoff "0.6"
# python transformer_debugger.py -expt_num "10.2.gate.5" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -load_weight_exp "10.2.gate.5" -load_weight_epoch 2 -gate_var_cutoff "0.8"


# python transformer_debugger.py -expt_num "10.3.gate.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3.gate.0" -load_weight_epoch 2 -gate_var_cutoff "0.05"
# python transformer_debugger.py -expt_num "10.3.gate.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3.gate.1" -load_weight_epoch 2 -gate_var_cutoff "0.1"
# python transformer_debugger.py -expt_num "10.3.gate.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3.gate.2" -load_weight_epoch 2 -gate_var_cutoff "0.2"
# python transformer_debugger.py -expt_num "10.3.gate.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3.gate.3" -load_weight_epoch 2 -gate_var_cutoff "0.4"
# python transformer_debugger.py -expt_num "10.3.gate.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3.gate.4" -load_weight_epoch 2 -gate_var_cutoff "0.6"
# python transformer_debugger.py -expt_num "10.3.gate.5" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -load_weight_exp "10.3.gate.5" -load_weight_epoch 2 -gate_var_cutoff "0.8"


# python transformer_debugger.py -expt_num "10.2.gate" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0 -gate_weight_exp "10.2" -gate_weight_epoch 2 -gate_var_cutoff "neg"
# python transformer_debugger.py -expt_num "10.3.gate" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0 -gate_weight_exp "10.3" -gate_weight_epoch 2 -gate_var_cutoff "neg"
# python transformer_debugger.py -expt_num "10.4.gate" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 100.0 -gate_weight_exp "10.4" -gate_weight_epoch 2 -gate_var_cutoff "neg"


# python transformer_debugger.py -expt_num "11.0" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 0.0 # 0.6k/topic *2
# python transformer_debugger.py -expt_num "11.1" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 1.0
# python transformer_debugger.py -expt_num "11.2" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 10.0
# python transformer_debugger.py -expt_num "11.3" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 50.0
# python transformer_debugger.py -expt_num "11.4" -num_samples 2000 -num_topic_samples 1200 -num_topics 10 -num_epochs 3 -transformer "bert-base-uncased" -l1_lambda 100.0


#Using the topic as binary feature [word present on not from atleast one of topic]
# python transformer_debugger.py -expt_num "ct3.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 10 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1
# #Counting the number of intersection too
# python transformer_debugger.py -expt_num "ct4.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 10 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1
# #Using the frequqncy of all the words in the topic as feature
# python transformer_debugger.py -expt_num "ct5.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 10 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1

#Adding the wordembedding assist for new words
# python transformer_debugger.py -expt_num "ct6.0" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 10
# python transformer_debugger.py -expt_num "ct6.1" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 20
# python transformer_debugger.py -expt_num "ct6.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 40
# python transformer_debugger.py -expt_num "ct6.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 80
# python transformer_debugger.py -expt_num "ct6.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 160

#Added domain specific topics

# python transformer_debugger.py -expt_num "ct7.1" -num_samples 1000 -num_topic_samples 600 -num_topics 11 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 20
# python transformer_debugger.py -expt_num "ct7.0" -num_samples 1000 -num_topic_samples 600 -num_topics 11 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 10

# python transformer_debugger.py -expt_num "ct8.0" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 10
# python transformer_debugger.py -expt_num "ct8.1" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 20


# python transformer_debugger.py -expt_num "ct8.2" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 
# python transformer_debugger.py -expt_num "ct8.3" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 10
# python transformer_debugger.py -expt_num "ct8.4" -num_samples 1000 -num_topic_samples 600 -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 20

# python transformer_debugger.py -expt_num "ct8.5" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 



# python transformer_debugger.py -expt_num "ct7.1" -num_samples 1000 -num_topic_samples 600 -num_topics 11 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 20
# python transformer_debugger.py -expt_num "ct7.1.1" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "word2vec-google-news-300" -num_neigh 20

# python transformer_debugger.py -expt_num "ct7.2" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 40
# python transformer_debugger.py -expt_num "ct7.3" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 80
# python transformer_debugger.py -expt_num "ct7.4" -num_samples 1000 -num_topic_samples 600 -num_topics 18 -num_epochs 50 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -emb_path "glove-wiki-gigaword-100" -num_neigh 160




#Experiment on the new toy dataset

python transformer_debugger.py -expt_num "ct9.0" -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.75 -spurious_ratio 0.90 

python transformer_debugger.py -expt_num "ct9.1" -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 

python transformer_debugger.py -expt_num "ct9.temp" -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 