#STAGE2-METHOD2 EXPT
mainepoch=8
main_mode="non_causal"
cbsize=1
max_len=80


transformer="bert-base-uncased"
lr=5e-5
batch_size=32
# batch_size=32
# lr=5e-3
hlayer=0



teloss_type="mse"
stage_mode="stage2_te_reg_strong"



#Dataset specific params
path="dataset/cebab_all/"
out_path="dataset/cebab_all"
dtype="cebab"
num_topics=4
topic_name="all"
sample="nosymm"


#Regularizing with true effect and then with DR for all the topic at once
for run_num in  1 2
do
    for cebab_all_ate_mode in "true" "de" "dr"
    do
        for te_lambda in  1 10
        do
            for pos_size in 1
            do
                for noise in 0.1 0.3
                do
                    for pval in "inf"
                    do
                        python transformer_debugger.py -expt_num "cad.cebabs2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($cebab_all_ate_mode).telambda($te_lambda)" -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -stage_mode $stage_mode -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -cebab_all_ate_mode $cebab_all_ate_mode #--symmetrize_main_cf
                    done
                done
            done
        done
    done
done