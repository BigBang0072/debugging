#STAGE2-METHOD2 EXPT
mainepoch=10
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
# stage_mode="stage2_te_reg_strong_with_width"
stage_mode="jtt_oversample"
debug_tidx=0


#Dataset specific parameters
path="dataset/twitter_aae_sentiment_race/"
out_path="dataset/twitter_aae_sentiment_race"
dtype="aae"
num_topics=1


topic_name="race" 
sample=10000




#13,14,15 with increased probabilty 0.99,0.30,0.70,0.01
#10,11,12 with previou probabiltiy 0.9,0.4,0.6,0.1
for s1epoch in 10 #5 10
do
    for run_num in 0 1 2 #14 15 #10 11 12 #1 2
    do
        for oslambda in 2 4 #2 6
        do
            for noise in 0.0
            do
                for pval in 0.5 0.6 0.7 0.8 0.9 0.99
                do
                    python transformer_debugger.py -expt_num "cad.aaejtt.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).s1epoch($s1epoch).oslambda($oslambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy"  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_pval $pval  -topic_name $topic_name -jtt_s1_epochs $s1epoch -os_lambda $oslambda  --bert_as_encoder -transformer $transformer --train_bert -l2_lambd 0.0 -gpu_num 1

                done
                #Waiting for the jobs to complete. Cannot handle so many parallel background jobs
                # for job in `jobs -p`
                # do
                #     wait $job
                # done
            done
        done
    done
done