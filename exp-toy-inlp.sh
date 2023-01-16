
model_type="nbow"
neg1_flip_method="remove_negation"
remmode="null_space"
main_mode="causal_same_sp"
advepoch=20
topicepoch=10
loss_type="x_entropy"
dropout_rate="0.0"
l2_lambd="0.0"
noise=0.0
hlayer=0
sample=1000
mainepoch=20
pval=0.8
run_num=1
path="dataset/nlp_toy2/"
out_path="dataset/nlp_toy2/"
hretrain="no_warm_encoder"


for noise in 0.1 0.3
do 
    for run_num in 1 2 3
    do
        for pval in 0.5 0.6 0.7 0.8 0.9 0.99
        do
            python transformer_debugger.py -expt_num "pt.inlp_hretrain.mt($model_type).neg1_fmethod($neg1_flip_method).remmode($remmode).mainmode($main_mode).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).drate($dropout_rate).l2($l2_lambd).n($noise).h($hlayer).s($sample).e($mainepoch).p($pval).r($run_num)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path $path -out_path $out_path -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $main_mode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -num_proj_iter $advepoch -topic_epochs $topicepoch -debug_tidx 1 -removal_mode $remmode -dtype "toynlp2"   -topic0_corr 1.0 -topic1_corr $pval -neg1_flip_method $neg1_flip_method --measure_flip_pdelta -run_num $run_num --inlp_train_main_head &
        done
    done
done 