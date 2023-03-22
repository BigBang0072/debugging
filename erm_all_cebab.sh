# STAGE2-ERM BASELINE
run_num=1
mainepoch=20
cbsize=1
main_mode="non_causal"
# noise=0.0
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
path="dataset/cebab_all/"
out_path="dataset/cebab_all"
dtype="cebab"
num_topics=4
topic_name="all"
sample="nosymm"




for run_num in 1 2
do
    for debug_tidx in 0
    do 
        for pos_size in 20
        do
            for noise in 0.1 0.3
            do 
                for pval in "inf"
                do
                    python transformer_debugger.py -expt_num "cad.cebabs2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).erm" -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -batch_size $batch_size -lr $lr  -dtype $dtype -loss_type "x_entropy"  -topic_pval $pval    -run_num $run_num -dropout_rate 0.0  -stage_mode $stage_mode --bert_as_encoder -transformer $transformer --train_bert #--symmetrize_main_cf
                done
            done 
        done
    done 
done
