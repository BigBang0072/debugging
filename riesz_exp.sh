#!/bin/bash
mkdir logs




run_num=1
mainepoch=12
hlayer=0
num_postalpha_layer=0
main_mode="non_causal"
cbsize=20
stage_mode="stage1_riesz"
noise=0.0


for debug_tidx in 0 1
do
    for sample in 1000
    do
        for rr_lambda in 1
        do
            for tmle_lambda in 1
            do
                for l2_lambda in 1
                do
                    for pvalt0 in 0.9
                    do
                        for pvalt1 in 0.5 0.6 0.7 0.8 0.9 0.99 
                        do
                            python transformer_debugger.py -expt_num "cad.s1riesz.rnum($run_num).pvalt0($pvalt0).pvalt1($pvalt1).dtidx($debug_tidx).rr_lmd($rr_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy"  -topic0_corr $pvalt0 -topic1_corr $pvalt1 -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -num_postalpha_layer $num_postalpha_layer
                        done
                    done
                done
            done
        done
    done
done