#!/bin/bash
mkdir logs



#STAGE2-METHOD1 EXPT
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
t0_ate="1.0"
t1_ate="0.0"
# ate_noise="0.1"
# inv_idx=0
closs_type="mse"
stage_mode="stage2_inv_reg"


for cont_lambda in 1 10 
do
    for sample in 1000
    do
        for pos_size in 5
        do
            for ate_noise in 1.0 0.7 0.5 0.2 0.0
            do
                for noise in 0.1 0.3
                do
                    for pval in 0.99 0.9 0.8 0.7 0.6 0.5
                    do
                        python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pval($pval).ate_noise($ate_noise).clambda($cont_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -closs_type $closs_type  -topic0_corr 1.0 -topic1_corr $pval -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -cont_lambda $cont_lambda -norm_lambda $norm_lambda   -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 1 -ate_noise $ate_noise -stage_mode $stage_mode
                    done
                done
            done
        done
    done
done




#STAGE2-METHOD2 EXPT
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
t0_ate="1.0"
t1_ate="0.0"
# ate_noise="0.1"
# inv_idx=0
teloss_type="mse"
stage_mode="stage2_te_reg"


for te_lambda in 1 10 0.1 0.01
do
    for sample in 1000
    do
        for pos_size in 5
        do
            for ate_noise in 1.0 0.7 0.5 0.2 0.0
            do
                for noise in 0.0 0.1 0.3
                do
                    for pval in 0.99 0.9 0.8 0.7 0.6 0.5
                    do
                        python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pval($pval).ate_noise($ate_noise).clambda($cont_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -teloss_type $teloss_type  -topic0_corr 1.0 -topic1_corr $pval -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -norm_lambda $norm_lambda   -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 1 -ate_noise $ate_noise -stage_mode $stage_mode
                    done
                done
            done
        done
    done
done



#STAGE2-METHOD1 EXPT
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
t0_ate="1.0"
t1_ate="0.0"
# ate_noise="0.1"
# inv_idx=0
closs_type="mse"
stage_mode="stage2_inv_reg"


for cont_lambda in 0.01 0.1
do
    for sample in 1000
    do
        for pos_size in 5
        do
            for ate_noise in 1.0 0.7 0.5 0.2 0.0
            do
                for noise in 0.0 0.1 0.3
                do
                    for pval in 0.99 0.9 0.8 0.7 0.6 0.5
                    do
                        python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pval($pval).ate_noise($ate_noise).clambda($cont_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -closs_type $closs_type  -topic0_corr 1.0 -topic1_corr $pval -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -cont_lambda $cont_lambda -norm_lambda $norm_lambda   -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 1 -ate_noise $ate_noise -stage_mode $stage_mode
                    done
                done
            done
        done
    done
done
