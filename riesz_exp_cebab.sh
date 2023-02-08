#!/bin/bash
mkdir logs




# run_num=6
mainepoch=10
# hlayer=0
num_postalpha_layer=0
main_mode="non_causal"
cbsize=1
stage_mode="stage1_riesz"
noise=0.0
batch_size=32
max_len=80

transformer="bert-base-uncased"
lr=5e-5

#Dataset specific parameters
path="dataset/cebab/"
out_path="dataset/cebab"
dtype="cebab"
topic_name="food"

for run_num in  1
do 
    for hlayer in 1 2
    do 
        for debug_tidx in 0
        do
            for sample in 750
            do
                for reg_lambda in 1 
                do 
                    for rr_lambda in 1
                    do
                        for tmle_lambda in 0
                        do
                            for l2_lambda in 0.0
                            do
                                python transformer_debugger.py -expt_num "cad.cebabs1riesz.rnum($run_num).sample($sample).hlayer($hlayer).pvaltsp($pvaltsp).dtidx($debug_tidx).rr_lmd($rr_lambda).reg_lmd($reg_lambda).tmle_lmd($tmle_lambda).l2_lmd($l2_lambda).dcf($dcf).noise($noise).topic($topic_name)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -dtype $dtype  -loss_type "x_entropy" -cfactuals_bsize $cbsize  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode -rr_lambda $rr_lambda -tmle_lambda $tmle_lambda -l2_lambd $l2_lambda -reg_lambda $reg_lambda -num_postalpha_layer $num_postalpha_layer -batch_size $batch_size -max_len $max_len --bert_as_encoder --train_bert -transformer $transformer -cebab_topic_name $topic_name
                                # -sp_topic_pval $pvaltsp
                                #--concat_word_emb  #--round_gval
                            done
                        done
                    done
                done
            done
        done
    done
done
