#!/bin/bash
mkdir logs




# run_num=6
mainepoch=200
# hlayer=0
num_postalpha_layer=0
main_mode="non_causal"
cbsize=1
stage_mode="stage1_riesz"
noise=0.0
batch_size=128
reg_mode="mse"

#Dataset specific parameters
path="dataset/nlp_toy3/"
out_path="dataset/nlp_toy3"
dtype="toynlp3"
num_topics=1
topic_name="causal" #causal or spurious --> based on them the dataset will be created




# for run_num in  15 16 17
# do 
#     for hlayer in 1 #1 2
#     do 
#         for debug_tidx in 0
#         do
#             for sample in 1000
#             do
#                 for reg_lambda in 1 
#                 do 
#                     for rr_lambda in 1
#                     do
#                         for tmle_lambda in 0
#                         do
#                             for l2_lambda in 0.0  0.1 1.0 10.0 100.0 200.0 1000.0
#                             do
#                                 for noise in 0.0
#                                 do 
#                                     for dcf in 1.0
#                                     do 
#                                         for pvaltsp in 0.5 0.6 0.7 0.8 0.9 0.99
#                                         do
#                                             python transformer_debugger.py -expt_num "cad.toy3s1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pvaltsp($pvaltsp).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype $dtype  -loss_type "x_entropy" -sp_topic_pval $pvaltsp -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len 20 -degree_confoundedness $dcf -topic_name $topic_name -riesz_reg_mode $reg_mode -best_gval_selection_metric "acc" --select_best_gval --select_best_alpha &  #--debug_mode #--concat_word_emb  #--round_gval
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


for run_num in  12 13 14 #6 7 8 
do 
    for hlayer in 1 #1 2
    do 
        for debug_tidx in 0
        do
            for sample in 1000
            do
                for reg_lambda in 1 
                do 
                    for rr_lambda in 1
                    do
                        for tmle_lambda in 0
                        do
                            for l2_lambda in 0.0 0.1 1.0 10.0 100.0 200.0 1000.0 #we dont need this when h=0
                            do
                                for noise in 0.0
                                do 
                                    for dcf in 1.0
                                    do 
                                        for pvaltsp in 0.7 #0.5 0.6 0.7 0.8 0.9 0.99
                                        do
                                            python transformer_debugger.py -expt_num "cad.toy3s1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pvaltsp($pvaltsp).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype $dtype  -loss_type "x_entropy" -sp_topic_pval $pvaltsp -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len 20 -degree_confoundedness $dcf -topic_name $topic_name -riesz_reg_mode $reg_mode -best_gval_selection_metric "loss" --select_best_gval --select_best_alpha & #--concat_word_emb  #--round_gval
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



# The experiemnt is for direct effect witou any rr component and any selection
# Last time we forgot to switch off the best gval selection 
# for run_num in  10 11 #10 11 #9 10 11 #9 10 11
# do 
#     for hlayer in 1 #1 2
#     do 
#         for debug_tidx in 0
#         do
#             for sample in 1000
#             do
#                 for reg_lambda in 1 
#                 do 
#                     for rr_lambda in 0
#                     do
#                         for tmle_lambda in 0
#                         do
#                             for l2_lambda in  0.0 1.0 10.0 100.0 200.0 1000.0 #we dont need this when h=0
#                             do
#                                 for noise in 0.0
#                                 do 
#                                     for dcf in 1.0
#                                     do 
#                                         for pvaltsp in 0.5 0.6 0.7 0.8 0.9 0.99
#                                         do
#                                             python transformer_debugger.py -expt_num "cad.toy3s1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pvaltsp($pvaltsp).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype $dtype  -loss_type "x_entropy" -sp_topic_pval $pvaltsp -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len 20 -degree_confoundedness $dcf -topic_name $topic_name -riesz_reg_mode $reg_mode & #&    # -best_gval_selection_metric "loss" & #--concat_word_emb  #--round_gval
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




# for run_num in  18 19 20 #0 1 2
# do 
#     for hlayer in 1 #1 2
#     do 
#         for debug_tidx in 0
#         do
#             for sample in 1000
#             do
#                 for reg_lambda in 1 
#                 do 
#                     for rr_lambda in 1
#                     do
#                         for tmle_lambda in 0
#                         do
#                             for l2_lambda in 0.0 0.1 1.0 10.0 100.0 200.0 1000.0
#                             do
#                                 for noise in 0.0
#                                 do 
#                                     for dcf in 1.0
#                                     do 
#                                         for pvaltsp in 0.5 0.6 0.7 0.8 0.9 0.99
#                                         do
#                                             python transformer_debugger.py -expt_num "cad.toy3s1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pvaltsp($pvaltsp).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr 5e-3 -dtype $dtype  -loss_type "x_entropy" -sp_topic_pval $pvaltsp -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len 20 -degree_confoundedness $dcf -topic_name $topic_name -riesz_reg_mode $reg_mode & #--concat_word_emb  #--round_gval
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