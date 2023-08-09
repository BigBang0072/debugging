#STAGE2-METHOD2 EXPT
mainepoch=5
main_mode="non_causal"
cbsize=1
max_len=20


# transformer="bert-base-uncased"
# lr=5e-5
# batch_size=32
batch_size=32
lr=5e-3
eiil_disc_lr=5
hlayer=1



# stage_mode="stage2_te_reg_strong_with_width"
stage_mode="eiil_irm"
debug_tidx=0


#Dataset specific parameters
path="dataset/mnist/"
out_path="dataset/mnist"
dtype="mnist"
num_topics=1
topic_name="rotation" 
sample=10000



#s1epoch 0 and eiil_disc_epoch 0 is pure IRM
#13,14,15 with increased probabilty 0.99,0.30,0.70,0.01
#10,11,12 with previou probabiltiy 0.9,0.4,0.6,0.1
for irm_lambda in 1e14 5e14  # 2 10
do 
    for run_num in 0 1 2 #14 15 #10 11 12 #1 2
    do
        for s1epoch in 0 #2 6
        do  
            for eiil_disc_epoch in 0 #5 10
            do
                for dcf in 0.0
                do 
                    for noise in 0.3
                    do
                        for pvaltsp in 0.5 0.6 0.7 0.8 0.9 0.99
                        do
                            python transformer_debugger.py -expt_num "cad.mnistirm.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pvaltsp($pvaltsp).dcf($dcf).s1epoch($s1epoch).eiilepoch($eiil_disc_epoch).irmlambda($irm_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy"  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx -stage_mode $stage_mode -sp_topic_pval $pvaltsp -degree_confoundedness $dcf  -topic_name $topic_name -jtt_s1_epochs $s1epoch -eiil_disc_epoch $eiil_disc_epoch -eiil_disc_lr $eiil_disc_lr -irm_lambda $irm_lambda  #--bert_as_encoder -transformer $transformer --train_bert

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
    done
done