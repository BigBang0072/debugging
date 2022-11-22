#!/bin/bash
mkdir logs




# STAGE2-ERM BASELINE
# run_num=1
# batch_size=32
# # sample=1000
# mainepoch=8
# # noise="0.0"
# hlayer=1
# main_mode="non_causal"
# # pval=0.5
# cbsize=20
# # pos_size=5
# # neg_size=5
# # cont_lambda="0.1"
# # norm_lambda="0.1"
# # t0_ate="1.0"
# # t1_ate="0.0"
# # ate_noise="0.1"
# # inv_idx=0
# # closs_type="mse"
# stage_mode="main"


# for sample in 1000
# do
#     for pos_size in 20
#     do
#         for noise in  0.0
#         do
#             for pvalt0 in  0.5 0.6 0.7 0.8 0.9 0.99 
#             do
#                 for pvalt1 in 0.8 0.9
#                 do
#                     python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pvalt0($pvalt0).pvalt1($pvalt1).erm.maintopic1" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy"  -topic0_corr $pvalt0 -topic1_corr $pvalt1 -cfactuals_bsize $cbsize    -run_num $run_num -dropout_rate 0.0 -debug_tidx 0  -stage_mode $stage_mode -batch_size $batch_size
#                 done
#             done
#         done
#     done
# done


# STAGE2-METHOD1 EXPT
# run_num=1
# batch_size=32
# # sample=1000
# mainepoch=8
# # noise="0.0"
# hlayer=1
# main_mode="non_causal"
# # pval=0.5
# cbsize=20
# # pos_size=5
# # neg_size=5
# # cont_lambda="0.1"
# # norm_lambda="1.0"
# t0_ate="0.0"
# t1_ate="1.0"
# # ate_noise="0.1"
# # inv_idx=0
# closs_type="mse"
# stage_mode="stage2_inv_reg"

# # IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
# #CHenge pval in topic corr and debugtidx and in the datahandle
# #Also change the ate value bc

# for lambda in 50,1 100,1 200,1
# do
#     IFS=',' read cont_lambda norm_lambda <<< "${lambda}"
#     for sample in 1000
#     do
#         for pos_size in 20
#         do
#             for ate_noise in 0.2 0.5 0.7
#             do
#                 for noise in 0.0
#                 do
#                     for pvalt0 in 0.99 0.9 0.8 0.7 0.6 0.5
#                     do
#                         for pvalt1 in 0.9
#                         do
#                             python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pvalt0($pvalt0).pvalt1($pvalt1).ate_noise($ate_noise).clambda($cont_lambda).maintopic1.poscf" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -closs_type $closs_type  -topic0_corr $pvalt0 -topic1_corr $pvalt1 -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -cont_lambda $cont_lambda -norm_lambda $norm_lambda   -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 0 -ate_noise $ate_noise -stage_mode $stage_mode -batch_size $batch_size
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done





# Stage 2 - Stratege 1 - Realistic noise (only added to the wrong topic as observed in stage 1 from riesz) + CF distribution has true counterfactuals
# run_num=1
# batch_size=32
# # sample=1000
# mainepoch=8
# # noise="0.0"
# hlayer=1
# main_mode="non_causal"
# # pval=0.5
# cbsize=20
# # pos_size=5
# # neg_size=5
# # cont_lambda="0.1"
# # norm_lambda="1.0"
# t0_ate="0.0"
# # t1_ate="1.0"
# # ate_noise="0.1"
# # inv_idx=0
# closs_type="mse"
# stage_mode="stage2_inv_reg"

# # IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
# #CHenge pval in topic corr and debugtidx and in the datahandle
# #Also change the ate value bc

# for lambda in 1,1 10,1 30,1 50,1 100,1 200,1
# do
#     IFS=',' read cont_lambda norm_lambda <<< "${lambda}"
#     for sample in 1000
#     do
#         for pos_size in 20
#         do
#             for ate_noise in 0.0 0.2 0.4 0.6 0.7
#             do
#                 for noise in 0.0
#                 do
#                     for pvalt0 in 0.99 0.9 0.8 0.7 0.6 0.5
#                     do
#                         for pvalt1andt1_ate in 0.9,0.7
#                         do
#                             IFS=',' read pvalt1 t1_ate <<< "${pvalt1andt1_ate}"
#                             python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pvalt0($pvalt0).pvalt1($pvalt1).t0ate($t0_ate).t1ate($t1_ate).ate_noise($ate_noise).clambda($cont_lambda).maintopic1.poscf.noisestage1" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -closs_type $closs_type  -topic0_corr $pvalt0 -topic1_corr $pvalt1 -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -cont_lambda $cont_lambda -norm_lambda $norm_lambda   -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 0 -ate_noise $ate_noise -stage_mode $stage_mode -batch_size $batch_size
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done






#STAGE2-METHOD2 EXPT
# run_num=1
# # sample=1000
# mainepoch=8
# # noise="0.0"
# hlayer=1
# main_mode="non_causal"
# # pval=0.5
# cbsize=20
# # pos_size=5
# # neg_size=5
# # cont_lambda="0.1"
# # norm_lambda="0.1"
# t0_ate="0.0"
# t1_ate="1.0"
# # ate_noise="0.1"
# # inv_idx=0
# teloss_type="mse"
# stage_mode="stage2_te_reg"


# for te_lambda in 100 1 10 200
# do
#     for sample in 1000
#     do
#         for pos_size in 20
#         do
#             for ate_noise in 0.2 0.5 0.7 1.0
#             do
#                 for noise in 0.0
#                 do
#                     for pvalt0 in 0.99 0.9 0.8 0.7 0.6 0.5
#                     do
#                         for pvalt1 in 0.9
#                         do
#                             python transformer_debugger.py -expt_num "cad.s2.rnum($run_num).noise($noise).pvalt0($pvalt0).pvalt1($pvalt1).ate_noise($ate_noise).telambda($te_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -teloss_type $teloss_type  -topic0_corr $pvalt0 -topic1_corr $pvalt1 -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 0 -ate_noise $ate_noise -stage_mode $stage_mode
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done



#STAGE2-METHOD2 EXPT + correct CF + correct noise
run_num=1
# sample=1000
mainepoch=8
# noise="0.0"
hlayer=1
main_mode="non_causal"
# pval=0.5
cbsize=20
# pos_size=5
# neg_size=5
# cont_lambda="0.1"
# norm_lambda="0.1"
t0_ate="0.0"
# t1_ate="1.0"
# ate_noise="0.1"
# inv_idx=0
teloss_type="mse"
stage_mode="stage2_te_reg_weak"


for te_lambda in 10 30 50 100 200
do
    for sample in 1000
    do
        for pos_size in 20
        do
            for ate_noise in 0.0 0.2 0.4 0.6 0.7
            do
                for noise in 0.0
                do
                    for pvalt0 in 0.99 0.9 0.8 0.7 0.6 0.5
                    do
                        for pvalt1andt1_ate in 0.9,0.7
                        do
                            IFS=',' read pvalt1 t1_ate <<< "${pvalt1andt1_ate}"
                            python transformer_debugger.py -expt_num "cad.s2m2weak.rnum($run_num).noise($noise).pvalt0($pvalt0).pvalt1($pvalt1).t0ate($t0_ate).t1ate($t1_ate).ate_noise($ate_noise).telambda($te_lambda)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -out_path "dataset/nlp_toy2" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype "toynlp2" -loss_type "x_entropy" -teloss_type $teloss_type  -topic0_corr $pvalt0 -topic1_corr $pvalt1 -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -t1_ate $t1_ate -debug_tidx 0 -ate_noise $ate_noise -stage_mode $stage_mode
                        done
                    done
                done
            done
        done
    done
done
