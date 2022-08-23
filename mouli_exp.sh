#!/bin/bash
mkdir logs


run_num=1
sample=1000
mainepoch=100
noise="0.0"
hlayer=1
main_mode="non_causal"
pval=0.5
cbsize=10
pos_size=5
neg_size=5
cont_lambda="0.1"
norm_lambda="0.1"
inv_idx=0
closs_type="mse"


#Starting the first experiment for mouli
python transformer_debugger.py -expt_num "cad.rnum($run_num)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -closs_type $closs_type  -topic0_corr 1.0 -topic1_corr $pval -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $neg_size -cont_lambda $cont_lambda -norm_lambda $norm_lambda -inv_idx $inv_idx  -run_num $run_num -dropout_rate 0.0