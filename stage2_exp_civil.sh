



#STAGE2-METHOD2 EXPT
mainepoch=20
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
debug_tidx=0



#Dataset specific params
path="dataset/cebab/"
out_path="dataset/cebab"
dtype="cebab"
num_topics=1



#Regularizing with true effect and then with DR
for topicANDsample in "food",350 #"food",750 "service",500 "ambiance",200 "noise",65
do 
    IFS=',' read topic_name sample <<< "${topicANDsample}"
    for run_num in 0 1 2
    do
        for te_lambda in  1 10
        do
            for pos_size in 1
            do
                for noise in 0.0
                do
                    for pvalANDt0_ate in 0.5,0.45 0.6,0.54 0.7,0.58 0.8,0.60 0.9,0.69   0.5,0.05 0.6,0.25 0.7,0.45 0.8,0.4 0.9,0.6
                    do
                        IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

                        python transformer_debugger.py -expt_num "cad.cebabs2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -cebab_topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert 
                    done
                done
            done
        done
    done
done




