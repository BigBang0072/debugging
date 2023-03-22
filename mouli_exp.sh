#!/bin/bash
mkdir logs


run_num=1
# sample=1000
mainepoch=15
# noise="0.0"
hlayer=1
main_mode="non_causal"
# pval=0.5
cbsize=20
# pos_size=5
# neg_size=5
cont_lambda="0.1"
norm_lambda="0.1"
# inv_idx=0
closs_type="mse"


for sample in 1000 10000
do
    for pos_size in 5 10 20
    do
        for noise in 0.0 0.1 0.3
        do
            for inv_idx in 0 1
            do 
                for pval in 0.5 0.6 0.7 0.8 0.9 0.99
                do
                    python transformer_debugger.py -expt_num "cad.rnum($run_num)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -closs_type $closs_type  -topic0_corr 1.0 -topic1_corr $pval -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -cont_lambda $cont_lambda -norm_lambda $norm_lambda -inv_idx $inv_idx  -run_num $run_num -dropout_rate 0.0
                done
            done
        done
    done
done