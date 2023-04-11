



#STAGE2-METHOD2 EXPT
mainepoch=1
main_mode="non_causal"
cbsize=1
max_len=80


transformer="bert-base-uncased"
lr=5e-5
batch_size=32
# batch_size=32
# lr=5e-3
hlayer=0



stage_mode="stage1_mouli"
debug_tidx=0



#Dataset specific parameters
path="dataset/twitter_aae_sentiment_race/"
out_path="dataset/twitter_aae_sentiment_race"
dtype="aae"
num_topics=1
replace_strategy="gpt3"
pos_size=1


for topicANDsample in "race",10000
do 
    IFS=',' read topic_name sample <<< "${topicANDsample}"
    for run_num in 0 1 2
    do
        for noise in 0.0
        do
            for pval in 0.5 0.6 0.7 0.8 0.9 0.95 0.99
            do
                for mvsel_mode in "loss" "acc"
                do 
                    python transformer_debugger.py -expt_num "cad.moulis1aae.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).mvsel($mvsel_mode)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -mouli_valid_sel_mode $mvsel_mode
                done
            done
        done
    done
done


