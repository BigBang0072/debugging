# STAGE2-ERM BASELINE
run_num=1
mainepoch=8
cbsize=1
main_mode="non_causal"
noise=0.0
max_len=80



transformer="bert-base-uncased"
lr=5e-5
batch_size=32
# batch_size=32
# lr=5e-3
hlayer=0


# pos_size=5
# neg_size=5
# cont_lambda="0.1"
# norm_lambda="0.1"
# t0_ate="1.0"
# t1_ate="0.0"
# ate_noise="0.1"
# inv_idx=0
# closs_type="mse"
stage_mode="main"


#Dataset specific params
path="dataset/civilcomments/"
out_path="dataset/civilcomments"
dtype="civilcomments"
num_topics=1



for topicANDsample in "race",5000 #"gender",5000 "religion",4000
do 
    IFS=',' read topic_name sample <<< "${topicANDsample}"
    for run_num in 0
    do
        for debug_tidx in 0
        do 
            for pval in  0.5 0.6 0.7 0.8 0.9 0.99
            do
                python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).pval($pval).erm" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -batch_size $batch_size -lr $lr  -dtype $dtype -loss_type "x_entropy" -topic_name $topic_name  -topic_pval $pval    -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx  -stage_mode $stage_mode --bert_as_encoder -transformer $transformer --train_bert 
            done
        done
    done 
done
