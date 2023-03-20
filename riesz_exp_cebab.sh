#!/bin/bash
mkdir logs




# run_num=6
mainepoch=20
# hlayer=0
num_postalpha_layer=0
main_mode="non_causal"
cbsize=1
stage_mode="stage1_riesz"
noise=0.0
max_len=80


transformer="bert-base-uncased"
lr=5e-5
batch_size=32
# batch_size=32
# lr=5e-3

#Dataset specific parameters
path="dataset/cebab/"
out_path="dataset/cebab"
dtype="cebab"
num_topics=1
# topic_name="food"

# for topicANDsample in "food",350 #"service",200 "ambiance",100 "noise",50
# for topicANDsample in  "food",1100 "ambiance",700 "service",900 "noise",500 #
for run_num in  14 15 # this is using the best alpha tracker #8 9 10
do 
    for topicANDsample in "food",500 "ambiance",340 "service",460 "noise",320 #this is with unk addition
    do 
        IFS=',' read topic_name sample <<< "${topicANDsample}"
        for reg_mode in  "mse" "bce" 
        do 
            for hlayer in 0
            do 
                for debug_tidx in 0
                do
                    for reg_lambda in 1 
                    do 
                        for rr_lambda in 1
                        do
                            for tmle_lambda in 0
                            do
                                for l2_lambda in 0.0 1.0 10.0 100.0 200.0 1000.0 #for toy3# 0.0001 0.001 0.01 0.1 
                                do
                                    for pval in "inf" #0.5 0.6 0.7 0.8 0.9 #None is natural correlation
                                    do 
                                        python transformer_debugger.py -expt_num "cad.cebabs1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pval($pval).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name).regmode($reg_mode)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -dtype $dtype  -loss_type "x_entropy" -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len $max_len -cebab_topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -riesz_reg_mode $reg_mode -best_gval_selection_metric "loss" --select_best_gval --select_best_alpha  #--symmetrize_main_cf #--separate_cebab_de  -num_sample_cebab_all 5000 #--only_de #change lr when using bert
                                        #--concat_word_emb  #--round_gval
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


for run_num in  17 18 # this is using the best alpha tracker #8 9 10
do 
    for topicANDsample in "food",500 "ambiance",340 "service",460 "noise",320 #this is with unk addition
    do 
        IFS=',' read topic_name sample <<< "${topicANDsample}"
        for reg_mode in  "mse" "bce" 
        do 
            for hlayer in 0
            do 
                for debug_tidx in 0
                do
                    for reg_lambda in 1 
                    do 
                        for rr_lambda in 1
                        do
                            for tmle_lambda in 0
                            do
                                for l2_lambda in 0.0 1.0 10.0 100.0 200.0 1000.0 #for toy3# 0.0001 0.001 0.01 0.1 
                                do
                                    for pval in "inf" #0.5 0.6 0.7 0.8 0.9 #None is natural correlation
                                    do 
                                        python transformer_debugger.py -expt_num "cad.cebabs1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pval($pval).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name).regmode($reg_mode)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -dtype $dtype  -loss_type "x_entropy" -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len $max_len -cebab_topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -riesz_reg_mode $reg_mode -best_gval_selection_metric "acc" --select_best_gval --select_best_alpha  #--symmetrize_main_cf #--separate_cebab_de  -num_sample_cebab_all 5000 #--only_de #change lr when using bert
                                        #--concat_word_emb  #--round_gval
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
