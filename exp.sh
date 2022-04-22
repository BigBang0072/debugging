
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

# python transformer_debugger.py -expt_num "ct9.0" -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.75 -spurious_ratio 0.90 

# python transformer_debugger.py -expt_num "ct9.1" -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 

# python transformer_debugger.py -expt_num "ct9.temp" -num_topics 10 -num_epochs 30 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 

# python transformer_debugger.py -expt_num "ct10.0" -num_topics 10 -num_epochs 15 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 -debug_cidx 1 -debug_tidx 6

# python transformer_debugger.py -expt_num "ct10.0.adv" -num_topics 10 -num_epochs 15 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 -debug_cidx 1 -debug_tidx 6 --reverse_grad

# python transformer_debugger.py -expt_num "ct10.0.adv" -num_topics 10 -num_epochs 15 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 -debug_cidx 1 -debug_tidx 0 --reverse_grad


#Making the current very toyish dataset more clear
# python transformer_debugger.py -expt_num "ct11.0" -num_topics 10 -num_epochs 70 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.85 -stage 1
# python transformer_debugger.py -expt_num "ct11.1" -num_topics 10 -num_epochs 70 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.88 -stage 1
# python transformer_debugger.py -expt_num "ct11.2" -num_topics 10 -num_epochs 70 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.91 -stage 1
# python transformer_debugger.py -expt_num "ct11.3" -num_topics 10 -num_epochs 70 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.94 -stage 1
# python transformer_debugger.py -expt_num "ct11.4" -num_topics 10 -num_epochs 70 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 0.97 -stage 1
# python transformer_debugger.py -expt_num "ct11.5" -num_topics 10 -num_epochs 70 -transformer "bert-base-uncased" -l1_lambda 0.0 -temb_dim 1 -path "dataset/nlp_toy/data/" -causal_ratio 0.85 -spurious_ratio 1.0 -stage 1


#New stage2 base using the NBOW
# python transformer_debugger.py -expt_num "ct12.main" -num_topics 10 -num_epochs 100 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 1.0 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 0 -lr 0.005
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 6 -lr 0.005 -num_proj_iter 10 -topic_epochs 0 #--extend_topic_set -num_neigh 10
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 0 -lr 0.005 -num_proj_iter 10 -topic_epochs 0
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 1 -lr 0.005 -num_proj_iter 10 -topic_epochs 0
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 2 -lr 0.005 -num_proj_iter 10 -topic_epochs 0
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 3 -lr 0.005 -num_proj_iter 10 -topic_epochs 0
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 6 -lr 0.005 -num_proj_iter 10 -topic_epochs 0
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 7 -lr 0.005 -num_proj_iter 10 -topic_epochs 0
# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 0 -lr 0.005 -num_proj_iter 10 -topic_epochs 0

# python transformer_debugger.py -expt_num "ct12.inlp" -num_topics 10 -num_epochs 30 -path "dataset/nlp_toy/data/" -emb_path "glove-wiki-gigaword-100" -causal_ratio 0.85 -spurious_ratio 0.90 -stage 2 --normalize_emb -debug_cidx 1 -debug_tidx 9 -lr 0.005 -num_proj_iter 10 -topic_epochs 0


# python transformer_debugger.py -expt_num "ct13.inlp" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 1.0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 5 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct13.inlp" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.95 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 5 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct13.inlp" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 5 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct13.inlp" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 5 -topic_epochs 10


#Here we captured the topic accuracy after the projection step by mistake
#But thye will also provide some useful information in terms of how much the topic0 is affecting the topic1
# python transformer_debugger.py -expt_num "ct14.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct14.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct14.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct14.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10


#Getting the topic accuracy before now
# python transformer_debugger.py -expt_num "ct15.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct15.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct15.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct15.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10


#Expt 16: #FP steps vs p : expectation: #FPSTEP decrese as p increase
# python transformer_debugger.py -expt_num "ct16.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct16.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.6 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct16.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct16.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.8 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct16.inlp.5" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.90 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct16.inlp.4" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.99 -main_topic 1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10


#Expt 17: #TP steps vs p: shoould be low for all
# python transformer_debugger.py -expt_num "ct17.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct17.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.6 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct17.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct17.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.8 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct17.inlp.5" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.90 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct17.inlp.4" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.99 -main_topic 0 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

#Now having a separate causal variable and multiple topics which are correlated with that.
#So that these topics could be used by the predictor in different proportions (mainly its just like having multiple correlated topics)
# python transformer_debugger.py -expt_num "ct18.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct18.inlp.4" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.5" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.6" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.7" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct18.inlp.8" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.9" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.10" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.11" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct18.inlp.12" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.13" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.14" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct18.inlp.15" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10



# python transformer_debugger.py -expt_num "ct19.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.4" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.5" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct19.inlp.6" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.7" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.8" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.9" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.10" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.11" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct19.inlp.12" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.13" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.14" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.15" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.16" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.17" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct19.inlp.18" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.19" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.20" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.21" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.22" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.23" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct19.inlp.24" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.25" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.26" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.27" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.28" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.29" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct19.inlp.30" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.31" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.32" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.33" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.34" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct19.inlp.35" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10


#Testing the main topic in presence of noise (now we dont have anything to get 100% accuracy)
# python transformer_debugger.py -expt_num "ct20.inlp.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.4" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.5" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.5 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct20.inlp.6" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.7" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.8" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.9" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.10" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.11" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.6 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct20.inlp.12" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.13" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.14" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.15" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.16" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.17" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.7 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct20.inlp.18" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.19" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.20" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.21" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.22" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.23" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.8 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct20.inlp.24" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.25" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.26" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.27" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.28" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.29" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 0.9 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10

# python transformer_debugger.py -expt_num "ct20.inlp.30" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.31" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.6 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.32" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.33" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.8 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.34" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epochs 10
# python transformer_debugger.py -expt_num "ct20.inlp.35" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 1.0 -main_topic -1 -stage 2 --normalize_emb -debug_tidx 0 -lr 0.005 -num_proj_iter 30 -topic_epoch 10





#Training the BOW on amazon dataset (BOW is not enough to train for this task)
# python transformer_debugger.py -expt_num "ct21.inlp.0" -num_sample 1000 -num_topics -1 -num_epochs 10 -path "dataset/amazon/" -emb_path "glove-wiki-gigaword-100"  -stage 2 --normalize_emb -debug_cidx 3 -debug_tidx 0 -lr 0.01 -num_proj_iter 30 -topic_epoch 10

#Training the transformer on the amazon dataset (changing the topic to remove)
#Why not just use the convergence angle to show if the topic is being used or not? Why remove stuffs?

#num topic epoch is also critical cuz if we are trining it wrong then could remove wrong stuff
# python transformer_debugger.py -expt_num "ct22.inlp.0" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 0 -lr 0.01  -num_epochs 1 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 20
# python transformer_debugger.py -expt_num "ct22.inlp.1" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 1 -lr 0.01  -num_epochs 3 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 20
# python transformer_debugger.py -expt_num "ct22.inlp.2" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 2 -lr 0.01  -num_epochs 3 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 20


# #Changing the number of dimension of the latent space (maybe smallar latent space are more sensitive to removal)
# python transformer_debugger.py -expt_num "ct23.inlp.0" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 0 -lr 0.01  -num_epochs 3 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 200
# python transformer_debugger.py -expt_num "ct23.inlp.1" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 1 -lr 0.01  -num_epochs 3 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 200
# python transformer_debugger.py -expt_num "ct23.inlp.2" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 2 -lr 0.01  -num_epochs 3 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 200

# script -c "/path/prog" /path/log.txt
# python transformer_debugger.py -expt_num "ct22.inlp.0" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 3 -debug_tidx 0 -lr 0.01  -num_epochs 5 -topic_epoch 3 -num_proj_iter 5 -hlayer_dim 768
#Training the movie domain [pos_adj,pos_gender,pos_movie] are active
# python transformer_debugger.py -expt_num "ct22.inlp.1" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 5 -debug_tidx 0 -lr 0.01  -num_epochs 10 -topic_epoch 4 -num_proj_iter 20 -hlayer_dim 768

# python transformer_debugger.py -expt_num "ct22.inlp.2" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 5 -debug_tidx 0 -lr 0.01  -num_epochs 10 -topic_epoch 4 -num_proj_iter 20 -hlayer_dim 768

# python transformer_debugger.py -expt_num "ct22.inlp.2.test" -num_sample 1000 -num_topics -1  -path "dataset/amazon/" -transformer "bert-base-uncased" -emb_path "glove-wiki-gigaword-100"  -stage 2  -debug_cidx 5 -debug_tidx 0 -lr 0.01  -num_epochs 10 -topic_epoch 4 -num_proj_iter 20 -hlayer_dim 768 --cached_bert










######################################################################################################
############################### ADVERSARIAL TRAINING ##########################################
######################################################################################################
#22nd March: Starting the convergence experiment
#Setup: nlp syn, crossentropy, relateive angle method to get convergence angle
#Expectation: As the correlation increases the converge angle between the classifier should decreased
# for a in 1 2 3 4 5 6 7 8 9
# do 
#     python transformer_debugger.py -expt_num "pt.rel.$a.0" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.1" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.6 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.2" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.3" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.8 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.4" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.5" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.99 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.6" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.999 -stage 2 --normalize_emb -lr 0.005
#     python transformer_debugger.py -expt_num "pt.rel.$a.7" -num_sample 1000 -num_topics 2 -num_epochs 10 -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9999 -stage 2 --normalize_emb -lr 0.005
# done


#Testing the effect of number of sample
# for r in 0 1 2 3 4
# do
#     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#     do
#         for e in 5 10 15
#         do
#             for s in 50 100 500 1000 10000
#             do
#                 for h in 0 1 5 10
#                 do
#                     python transformer_debugger.py -expt_num "pt.rel.h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005
#                 done
#             done       
#         done
#     done
# done

# for r in 0
# do
#     for n in 0.025 0.05 0.1 0.0
#     do
#         for p in 0.5 0.6 0.7 0.8 0.9 0.99
#         do
#             for e in 10
#             do
#                 for s in 500
#                 do
#                     for h in 0
#                     do
#                         python transformer_debugger.py -expt_num "pt.rel.n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005
#                     done
#                 done
#             done       
#         done
#     done
# done


#Testing the effect of number of epoch on convergence angle
# for l in 5 15
# do
#     for a in 0 1 3
#     do 
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.0" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.5 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.1" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.6 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.2" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.7 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.3" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.8 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.4" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.5" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.99 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.6" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.999 -stage 2 --normalize_emb -lr 0.005
#         python transformer_debugger.py -expt_num "pt.rel.$l.$a.7" -num_sample 1000 -num_topics 2 -num_epochs $l -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr 0.9999 -stage 2 --normalize_emb -lr 0.005
#     done
# done


# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for l2_lambd in 0.0001 0.001 0.01 0.1 1.0 10.0
#             do
#                 for e in 5 10 20
#                 do
#                     for d in "non_causal"
#                     do
#                         for h in 0
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 0.005 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done


# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for l2_lambd in 0.0001 0.0002 0.0004 0.0006 0.0008 0.001
#             do
#                 for e in 10
#                 do
#                     for d in "non_causal"
#                     do
#                         for h in 5
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 0.005 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done


# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for l2_lambd in 0.00001 0.00002 0.00004 0.00006 0.00008 0.0001
#             do
#                 for e in 10
#                 do
#                     for d in "non_causal"
#                     do
#                         for h in 10
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 0.005 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

#Training corss entropy regularized
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for e in 15
#             do
#                 for l2_lambd in 0.0 0.00001 0.0001 0.001 0.01 0.1
#                 do
#                     for d in "non_causal"
#                     do
#                         for h in 0 1 5 10
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# Training the dropout -regularized x-entropy loss
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for dropout_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#             do
#                 for l2_lambd in 0.0
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for h in 0 1 5 10
#                             do
#                                 for s in 500
#                                 do
#                                     for n in 0.0
#                                     do
#                                         for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                         do
#                                             python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                         done
#                                     done
#                                 done
#                             done       
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

#Variation with sample
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDdr in 1,0.9 5,0.6 10,0.4
#             do
#                 IFS=',' read h dropout_rate <<< "${hANDdr}"
#                 for l2_lambd in 0.0
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 100 500 1000
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# #Variation with epochs
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDdr in 1,0.9 5,0.6 10,0.4
#             do
#                 IFS=',' read h dropout_rate <<< "${hANDdr}"
#                 for l2_lambd in 0.0
#                 do
#                     for e in 10 15 20
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# #Variaiton with noise
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDdr in 1,0.9 5,0.6 10,0.4
#             do
#                 IFS=',' read h dropout_rate <<< "${hANDdr}"
#                 for l2_lambd in 0.0
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0 0.025 0.05 0.1
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done


# #Training the dropout regularization with max-margin
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for dropout_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#             do
#                 for l2_lambd in 0.0
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for h in 1 5 10
#                             do
#                                 for s in 500
#                                 do
#                                     for n in 0.0
#                                     do
#                                         for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                         do
#                                             python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                         done
#                                     done
#                                 done
#                             done       
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



#Variation with sample size
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDdr in 1,0.9 5,0.6 10,0.4
#             do
#                 IFS=',' read h dropout_rate <<< "${hANDdr}"
#                 for l2_lambd in 0.0
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 100 500 1000
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done





# #Variation with epochs
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDdr in 1,0.9 5,0.6 10,0.4
#             do
#                 IFS=',' read h dropout_rate <<< "${hANDdr}"
#                 for l2_lambd in 0.0
#                 do
#                     for e in 10 15 20
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done



# #Variaiton with noise
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDdr in 1,0.9 5,0.6 10,0.4
#             do
#                 IFS=',' read h dropout_rate <<< "${hANDdr}"
#                 for l2_lambd in 0.0
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0 0.025 0.05 0.1
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp"
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done



# for both svm and x_entropy we get the same lambda for f=diff layer
# so use the below template for both

#Training the SVM
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for l2_lambd in 0.0 0.0001 0.001 0.01 0.1
#             do
#                 for e in 10 15 20
#                 do
#                     for d in "non_causal"
#                     do
#                         for h in 0
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type
#                                     done
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done


#Teainign the model withoug any regularization (most relaxed)
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.0 1,0.0 5,0.0 10,0.0
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 15
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 100 500 1000
#                         do
#                             for n in 0.0
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# #Getting variation with epochs
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.0 1,0.0 5,0.0 10,0.0
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 10 15 20
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 500
#                         do
#                             for n in 0.0
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# # #Getting variation with noise
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.0 1,0.0 5,0.0 10,0.0
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 15
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 500
#                         do
#                             for n in 0.0 0.025 0.05 0.1
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done


#Changelog:
# making the debug tidx to be 0 and the main topic to be topic 1 and correlation vary in tpoic 0

# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for l2_lambd in 0.0 0.00001 0.0001 0.001 0.01 0.1
#             do
#                 for hlayer in 0 1 5 10
#                 do
#                     for e in 15
#                     do
#                         for d in "non_causal"
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).pt0(1.0).pt1($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr $p -topic1_corr 1.0 -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                     done
#                                 done
#                             done       
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.00001
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 15
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 100 500 1000
#                         do
#                             for n in 0.0
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).pt0($p).pt1(1.0).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr $p -topic1_corr 1.0 -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# # # #Getting variation with epochs
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.00001
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 10 15 20
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 500
#                         do
#                             for n in 0.0
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).pt0($p).pt1(1.0).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr $p -topic1_corr 1.0 -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# # # #Getting variation with noise
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.00001
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 15
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 500
#                         do
#                             for n in 0.0 0.025 0.05 0.1
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).pt0($p).pt1(1.0).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr $p -topic1_corr 1.0 -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

#Getting variation with sample size
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.0001
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 15
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 100 500 1000
#                         do
#                             for n in 0.0
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# # #Getting variation with epochs
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.0001
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 10 15 20
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 500
#                         do
#                             for n in 0.0
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done

# # #Getting variation with noise
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for r in 0
#         do
#             for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.0001
#             do
#                 IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                 for e in 15
#                 do
#                     for d in "non_causal"
#                     do
#                         for s in 500
#                         do
#                             for n in 0.0 0.025 0.05 0.1
#                             do
#                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                 do
#                                     python transformer_debugger.py -expt_num "pt.rel.lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($hlayer).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $hlayer -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype "nlp"
#                                 done
#                             done
#                         done       
#                     done
#                 done
#             done
#         done
#     done
# done






#Testing on the TABULAR data
#We will only test in h=0 here since only here the spurious and causal make sense
# for dtype in "tabular"
# do
#     for inv_dims in 1 10 100
#     do
#         for loss_type in "linear_svm"
#         do
#             for hretrain in "no_warm_encoder"
#             do
#                 for r in 0
#                 do
#                     for l2_lambd in 0.0 0.00001 0.0001 0.001 0.01 0.1
#                     do
#                         for e in 15
#                         do
#                             for d in "non_causal"
#                             do
#                                 for h in 0
#                                 do
#                                     for s in 500 1000
#                                     do
#                                         for n in 0.0
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.dtype($dtype).invdims($inv_dims).lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype $dtype -inv_dims $inv_dims
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#Varying the number of samples
# for dtype in "tabular"
# do
#     for tab_sigma_ubound in 0.1
#     do
#         for inv_dimsANDl2 in 100,200.0
#         do
#             IFS=',' read inv_dims l2_lambd <<< "${inv_dimsANDl2}"
#             for loss_type in "linear_svm"
#             do
#                 for hretrain in "no_warm_encoder"
#                 do
#                     for r in 0
#                     do
#                         for e in 15
#                         do
#                             for d in "non_causal"
#                             do
#                                 for h in 0
#                                 do
#                                     for s in 100 500 1000
#                                     do
#                                         for n in 0.0
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.sigma_ubound($tab_sigma_ubound).dtype($dtype).invdims($inv_dims).lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype $dtype -inv_dims $inv_dims -tab_sigma_ubound $tab_sigma_ubound
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# #Varying the number of epoch
# for dtype in "tabular"
# do
#     for tab_sigma_ubound in 0.1
#     do
#         for inv_dimsANDl2 in 100,200.0
#         do
#             IFS=',' read inv_dims l2_lambd <<< "${inv_dimsANDl2}"
#             for loss_type in "linear_svm"
#             do
#                 for hretrain in "no_warm_encoder"
#                 do
#                     for r in 0
#                     do
#                         for e in 10 15 20
#                         do
#                             for d in "non_causal"
#                             do
#                                 for h in 0
#                                 do
#                                     for s in 500
#                                     do
#                                         for n in 0.0
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.sigma_ubound($tab_sigma_ubound).dtype($dtype).invdims($inv_dims).lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype $dtype -inv_dims $inv_dims -tab_sigma_ubound $tab_sigma_ubound
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#Varying the lambda to higher value
# for dtype in "tabular"
# do
#     for tab_sigma_ubound in 0.1
#     do
#         for inv_dims in 10 100
#         do
#             for loss_type in "linear_svm"
#             do
#                 for hretrain in "no_warm_encoder"
#                 do
#                     for r in 0
#                     do
#                         for l2_lambd in 500 1000 2000
#                         do
#                             for e in 15
#                             do
#                                 for d in "non_causal"
#                                 do
#                                     for h in 0
#                                     do
#                                         for s in 500
#                                         do
#                                             for n in 0.0
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.sigma_ubound($tab_sigma_ubound).dtype($dtype).invdims($inv_dims).lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype $dtype -inv_dims $inv_dims -tab_sigma_ubound $tab_sigma_ubound
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

#Varying the noise level
# for dtype in "tabular"
# do
#     for tab_sigma_ubound in 0.1
#     do
#         for inv_dimsANDl2 in 100,200.0
#         do
#             IFS=',' read inv_dims l2_lambd <<< "${inv_dimsANDl2}"
#             for loss_type in "linear_svm"
#             do
#                 for hretrain in "no_warm_encoder"
#                 do
#                     for r in 0
#                     do
#                         for e in 15
#                         do
#                             for d in "non_causal"
#                             do
#                                 for h in 0
#                                 do
#                                     for s in 500
#                                     do
#                                         for n in 0.0 0.025 0.05 0.1
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.sigma_ubound($tab_sigma_ubound).dtype($dtype).invdims($inv_dims).lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype $dtype -inv_dims $inv_dims -tab_sigma_ubound $tab_sigma_ubound
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



# #Getting the sigma ubound
# for dtype in "tabular"
# do
#     for tab_sigma_ubound in 0.01 0.1 1.0 1.5 
#     do
#         for inv_dimsANDl2 in 100,200.0
#         do
#             IFS=',' read inv_dims l2_lambd <<< "${inv_dimsANDl2}"
#             for loss_type in "linear_svm"
#             do
#                 for hretrain in "no_warm_encoder"
#                 do
#                     for r in 0
#                     do
#                         for e in 15
#                         do
#                             for d in "non_causal"
#                             do
#                                 for h in 0
#                                 do
#                                     for s in 500
#                                     do
#                                         for n in 0.0
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.sigma_ubound($tab_sigma_ubound).dtype($dtype).invdims($inv_dims).lt($loss_type).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dtype $dtype -inv_dims $inv_dims -tab_sigma_ubound $tab_sigma_ubound
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done





# for model_type in "bert-large-uncased"
# do
#     for loss_type in "x_entropy"
#     do
#         for hretrain in "no_warm_encoder"
#         do
#             for r in 0
#             do
#                 for dropout_rate in 0.0
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for e in 3
#                         do
#                             for d in "non_causal"
#                             do
#                                 for h in 0
#                                 do
#                                     for s in 1000
#                                     do
#                                         for n in 0.0
#                                         do
#                                             for p in 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 1 -num_epochs $e -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-6 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#Training the bert model
for model_type in "bert-base-uncased"
do
    for loss_type in "x_entropy"
    do
        for hretrain in "no_warm_encoder"
        do
            for r in 0
            do
                for dropout_rate in 0.0
                do
                    for l2_lambd in 0.0
                    do
                        for e in 3
                        do
                            for d in "non_causal"
                            do
                                for h in 0
                                do
                                    for s in 10000
                                    do
                                        for n in 0.0
                                        do
                                            for p in 0.99 0.9 0.8 0.7 0.6 0.5
                                            do
                                                python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 1 -num_epochs $e -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -neg1_flip_method "dont_measure" --measure_flip_pdelta
                                            done
                                        done
                                    done
                                done       
                            done
                        done
                    done
                done
            done
        done
    done
done





######################################################################################################
############################### ADVERSARIAL TRAINING ##########################################
######################################################################################################


#Starting the adversarial training
# for r in 0
# do
#     for e in 1 5
#     do
#         for g in 1 10
#         do
#             for a in 5 10 20
#             do
#                 for h in 0 1 5 20
#                 do
#                     for s in  500 1000 100
#                     do
#                         for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                         do
#                             python transformer_debugger.py -expt_num "pt.rel.g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005 -adv_rm_epochs $a -rev_grad_strength $g -debug_tidx 1
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done




# for r in 0 1 2
# do
#     for e in 1
#     do
#         for g in 1
#         do
#             for a in 10
#             do
#                 for h in 0 1 2 3
#                 do
#                     for s in 500
#                     do
#                         for p in 0.6 0.7 0.8 0.9 0.99
#                         do
#                             python transformer_debugger.py -expt_num "pt.rel.g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005 -adv_rm_epochs $a -rev_grad_strength $g -debug_tidx 1
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done


# #Testing the epoch variance
# for r in 0 1 2
# do
#     for e in 1
#     do
#         for g in 1
#         do
#             for a in 10
#             do
#                 for h in 0
#                 do
#                     for s in  100 500 1000
#                     do
#                         for p in 0.6 0.7 0.8 0.9 0.99
#                         do
#                             python transformer_debugger.py -expt_num "pt.rel.g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005 -adv_rm_epochs $a -rev_grad_strength $g -debug_tidx 1
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done

# #Testing the epoch variance
# for r in 0 1 2
# do
#     for e in 1
#     do
#         for g in 1
#         do
#             for a in 5 10 15 20
#             do
#                 for h in 0
#                 do
#                     for s in  500
#                     do
#                         for p in 0.6 0.7 0.8 0.9 0.99
#                         do
#                             python transformer_debugger.py -expt_num "pt.rel.g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005 -adv_rm_epochs $a -rev_grad_strength $g -debug_tidx 1
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done

#Testing the noise variance
# for r in 0
# do
#     for e in 20
#     do
#         for g in 1
#         do
#             for a in 20
#             do
#                 for h in 1 5 0
#                 do
#                     for s in  500
#                     do
#                         for n in 0.025 0.05 0.1 0.0
#                         do
#                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                             do
#                                 python transformer_debugger.py -expt_num "pt.rel.n($n).g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 --normalize_emb -lr 0.005 -adv_rm_epochs $a -rev_grad_strength $g -debug_tidx 1
#                             done
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done


#MAX-MARGIN based adversarial training
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0 0.025 0.05 0.1
#             do
#                 for sample in 500 1000 100
#                 do
#                     for l2_lambd in 0.0 0.00001 0.0001 0.001 0.01
#                     do
#                         for mainepoch in 20
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for hlayer in 0 1 2 3 5
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hlayer in 0 1 2 4
#                     do
#                         for l2_lambd in 0.0 0.00001 0.0001 0.001 0.01 0.1
#                         do
#                             for mainepoch in 1
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

#Getting the sample size
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 100 500 1000
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 2,0.001 4,0.001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#Varying the noise level
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0 0.025 0.05 0.1
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 2,0.001 4,0.001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

#Varying the removal epich
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 2,0.001 4,0.001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 5 10 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#Varying the gradient remova sterength
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 2,0.001 4,0.001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1 5 10
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#VArying the main epoch
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 2,0.001 4,0.001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1 10 20
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

#Choose the correct lambda and then test the sample,epoch effect.
#But is it correct to do the selection of lanmda which is most hurtful?



#Starting the cross entorpy based adversarial removal but with regularization
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for dropout_rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#                         do
#                             for mainepoch in 1
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for hlayer in 0 1 2 4
#                                     do
#                                         for advepoch in 20
#                                         do
#                                             for grstrength in 1
#                                             do
#                                                 for remmode in "adversarial"
#                                                 do
#                                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                     do
#                                                         python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp" -dropout_rate $dropout_rate
#                                                     done
#                                                 done
#                                             done
#                                         done       
#                                     done
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

#Varying the sample size
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 100 500 1000
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for hANDdrate in 1,0.9 2,0.8 4,0.7
#                         do
#                             IFS=',' read hlayer dropout_rate <<< "${hANDdrate}"
#                             for mainepoch in 1
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp" -dropout_rate $dropout_rate
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Varying the removal epochs
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for hANDdrate in 1,0.9 2,0.8 4,0.7
#                         do
#                             IFS=',' read hlayer dropout_rate <<< "${hANDdrate}"
#                             for mainepoch in 1
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for advepoch in 5 10 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp" -dropout_rate $dropout_rate
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Varying the noise ratio
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0 0.025 0.05 0.1
#             do
#                 for sample in 500
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for hANDdrate in 1,0.9 2,0.8 4,0.7
#                         do
#                             IFS=',' read hlayer dropout_rate <<< "${hANDdrate}"
#                             for mainepoch in 1
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp" -dropout_rate $dropout_rate
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Varying the reversal strength
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for hANDdrate in 1,0.9 2,0.8 4,0.7
#                         do
#                             IFS=',' read hlayer dropout_rate <<< "${hANDdrate}"
#                             for mainepoch in 1
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1 5 25
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp" -dropout_rate $dropout_rate
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Varying the main epochs
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for l2_lambd in 0.0
#                     do
#                         for hANDdrate in 1,0.9 2,0.8 4,0.7
#                         do
#                             IFS=',' read hlayer dropout_rate <<< "${hANDdrate}"
#                             for mainepoch in 1 10 20
#                             do
#                                 for mainmode in "non_causal"
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp" -dropout_rate $dropout_rate
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done




#Starting the cross entorpy based adversarial removal but with regularization
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for l2_lambd in 0.0 0.00001 0.0001 0.001 0.01 0.1
#                     do
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for hlayer in 0 1 2 4
#                                 do
#                                     for advepoch in 20
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "adversarial"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#Sample variaiton
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 100 500 1000
#                 do
#                     for hANDl2 in 0,0.0 1,0.0 2,0.0 4,0.0
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Getting the noise variation
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0 0.025 0.05 0.1
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.0 1,0.0 2,0.0 4,0.0
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Getting the adv epoch variation
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.0 1,0.0 2,0.0 4,0.0
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 5 10 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Getting the removal strength variation
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.0 1,0.0 2,0.0 4,0.0
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1 5 25
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



# #Later also hobserve the effect of main_epoch variaiton cuz that will be responsible for bringing the topic info upfront
# for loss_type in "x_entropy"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.0 1,0.0 2,0.0 4,0.0
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 1 10 20
#                         do
#                             for mainmode in "non_causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for grstrength in 1
#                                     do
#                                         for remmode in "adversarial"
#                                         do
#                                             for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                             do
#                                                 python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -dtype "nlp"
#                                             done
#                                         done
#                                     done
#                                 done       
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



#Training the BERT model for the adversarial removal
for model_type in "bert-base-uncased"
do
    for loss_type in "x_entropy"
    do
        for hretrain in "no_warm_encoder"
        do
            for run in 0
            do
                for noise in 0.0
                do
                    for sample in 10000
                    do
                        for l2_lambd in 0.0
                        do
                            for dropout_rate in 0.0
                            do
                                for mainepoch in 0
                                do
                                    for mainmode in "non_causal"
                                    do
                                        for hlayer in 0
                                        do
                                            for advepoch in 5
                                            do
                                                for grstrength in 1.0
                                                do
                                                    for remmode in "adversarial"
                                                    do
                                                        for adv_rm_method in "adv_rm_with_main"
                                                        do
                                                            for p in 0.99 0.9 0.8 0.7 0.6 0.5
                                                            do
                                                                python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).remmode($remmode).adv_rm_method($adv_rm_method).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).noise($noise).hlayer($hlayer).sample($sample).p($p).run($run)" -num_sample $sample -num_topics 1 -num_epochs $mainepoch -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 0 -removal_mode $remmode -dropout_rate $dropout_rate -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -adv_rm_method $adv_rm_method -neg1_flip_method "dont_measure" --measure_flip_pdelta
                                                            done
                                                        done
                                                    done
                                                done       
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done








#####################################################################################################
################################         NULL SPACE REMOVAL        ##################################
#####################################################################################################
#Training a pure causal classifier
# for r in 0
# do
#     for e in 20
#     do
#         for g in 1
#         do
#             for t in 10
#             do
#                 for a in 20
#                 do
#                     for d in "causal" "non_causal"
#                     do
#                         for h in 0
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.0 0.025 0.05 0.1
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         for m in "null_space"
#                                         do
#                                             python transformer_debugger.py -expt_num "pt.rel.d($d).m($m).t($t).n($n).g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -removal_mode $m -main_model_mode $d --normalize_emb -lr 0.005 -num_proj_iter $a -topic_epochs $t  -rev_grad_strength $g -debug_tidx 1
#                                         done
#                                     done
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done





# for r in 0
# do
#     for e in 20
#     do
#         for g in 1
#         do
#             for t in 10
#             do
#                 for a in 20
#                 do
#                     for d in "causal" "non_causal"
#                     do
#                         for h in 0
#                         do
#                             for s in 100 500 1000
#                             do
#                                 for n in 0.025
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         for m in "null_space"
#                                         do
#                                             python transformer_debugger.py -expt_num "pt.rel.d($d).m($m).t($t).n($n).g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -removal_mode $m -main_model_mode $d --normalize_emb -lr 0.005 -num_proj_iter $a -topic_epochs $t  -rev_grad_strength $g -debug_tidx 1
#                                         done
#                                     done
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done


# for r in 0
# do
#     for e in 20
#     do
#         for g in 1
#         do
#             for t in 10
#             do
#                 for a in 20
#                 do
#                     for d in "causal" "non_causal"
#                     do
#                         for h in 0 1 5 10
#                         do
#                             for s in 500
#                             do
#                                 for n in 0.025
#                                 do
#                                     for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                     do
#                                         for m in "null_space"
#                                         do
#                                             python transformer_debugger.py -expt_num "pt.rel.d($d).m($m).t($t).n($n).g($g).a($a).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 2 -num_epochs $e -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $n -num_hidden_layer $h -stage 2 -removal_mode $m -main_model_mode $d --normalize_emb -lr 0.005 -num_proj_iter $a -topic_epochs $t  -rev_grad_strength $g -debug_tidx 1
#                                         done
#                                     done
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done       
#         done
#     done
# done


#Satrting the max-margin null-sapce iteration
#VArying the number of samples
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 100 500 1000
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.0001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 20
#                         do
#                             for mainmode in "causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for topicepoch in 10
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "null_space"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -num_proj_iter $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -topic_epochs $topicepoch
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


# #Varying the noise
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0 0.025 0.05 0.1
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.0001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 20
#                         do
#                             for mainmode in "causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for topicepoch in 10
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "null_space"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -num_proj_iter $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -topic_epochs $topicepoch
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done


#VAruing the main epoch and then sample variation
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0
#             do
#                 for sample in 1000
#                 do
#                     for hANDl2 in 1,0.01 5,0.001 10,0.0001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 10
#                         do
#                             for mainmode in "causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for topicepoch in 10
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "null_space"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -num_proj_iter $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -topic_epochs $topicepoch
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done

# #Noise variaiton in the reduced main epoch
# for loss_type in "linear_svm"
# do
#     for hretrain in "no_warm_encoder"
#     do
#         for run in 0
#         do
#             for noise in 0.0 0.025 0.05 0.1
#             do
#                 for sample in 500
#                 do
#                     for hANDl2 in 0,0.01 1,0.01 5,0.001 10,0.0001
#                     do
#                         IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
#                         for mainepoch in 10
#                         do
#                             for mainmode in "causal"
#                             do
#                                 for advepoch in 20
#                                 do
#                                     for topicepoch in 10
#                                     do
#                                         for grstrength in 1
#                                         do
#                                             for remmode in "null_space"
#                                             do
#                                                 for p in 0.5 0.6 0.7 0.8 0.9 0.99
#                                                 do
#                                                     python transformer_debugger.py -expt_num "pt.rel.remmode($remmode).grstrength($grstrength).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/data/" -emb_path "glove-wiki-gigaword-100" -topic0_corr 1.0 -topic1_corr $p -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -num_proj_iter $advepoch -rev_grad_strength $grstrength -debug_tidx 1 -removal_mode $remmode -topic_epochs $topicepoch
#                                                 done
#                                             done
#                                         done
#                                     done       
#                                 done
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



#Training the null-space model for the BERT
for model_type in "bert-base-uncased"
do
    for loss_type in "x_entropy"
    do
        for hretrain in "no_warm_encoder"
        do
            for run in 0
            do
                for noise in 0.0
                do
                    for sample in 10000
                    do
                        for hANDl2 in 0,0.0
                        do
                            IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
                            for mainepoch in 3
                            do
                                for mainmode in "causal"
                                do
                                    for advepoch in 7
                                    do
                                        for topicepoch in 1
                                        do
                                            for remmode in "null_space"
                                            do
                                                for p in 0.99 0.9 0.8 0.7 0.6 0.5
                                                do
                                                    python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).remmode($remmode).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 1 -num_epochs $mainepoch -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -num_proj_iter $advepoch  -debug_tidx 0 -removal_mode $remmode -topic_epochs $topicepoch -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -neg1_flip_method "dont_measure" --measure_flip_pdelta
                                                done
                                            done
                                        done
                                    done       
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done









