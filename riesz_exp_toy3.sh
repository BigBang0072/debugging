#!/bin/bash
mkdir logs




run_num=2
mainepoch=50
hlayer=1
num_postalpha_layer=1
main_mode="non_causal"
cbsize=1
stage_mode="stage1_riesz"
noise=0.0

#Dataset specific parameters
path="dataset/nlp_toy3/"
out_path="dataset/nlp_toy3"
dtype="toynlp3"

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
                    for pvaltsp in 0.5 0.6 0.7 0.8 0.9 0.99
                    do
                        python transformer_debugger.py -expt_num "cad.toy3s1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pvaltsp($pvaltsp).dtidx($debug_tidx).rr_lmd($rr_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype $dtype  -loss_type "x_entropy" -sp_topic_pval $pvaltsp -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -num_postalpha_layer $num_postalpha_layer &
                    done
                done
            done
        done
    done
done