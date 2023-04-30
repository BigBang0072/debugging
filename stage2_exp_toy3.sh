#STAGE2-METHOD2 EXPT
mainepoch=20
main_mode="non_causal"
cbsize=1
max_len=20


# transformer="bert-base-uncased"
# lr=5e-5
# batch_size=32
batch_size=32
lr=5e-3
hlayer=1



teloss_type="mse"
# stage_mode="stage2_te_reg_strong_with_width"
stage_mode="stage2_te_reg_strong"
debug_tidx=0


#Dataset specific parameters
path="dataset/nlp_toy3/"
out_path="dataset/nlp_toy3"
dtype="toynlp3"
num_topics=1
topic_name="spurious" #causal or spurious --> based on them the dataset will be created
sample=1000




#13,14,15 with increased probabilty 0.99,0.30,0.70,0.01
#10,11,12 with previou probabiltiy 0.9,0.4,0.6,0.1
for run_num in 13 14 15 #14 15 #10 11 12 #1 2
do
    for dcf in 1.0
    do 
        for hwidth in 0.0 #DONT not registered as fname #0.0 #0.05 0.1 0.2 0.5 1.0
        do 
            for te_lambda in  0 1 10 100 1000 4000 10000 #100 4000 #10 100 1000 4000 10000 #0 10 100 1000 4000 10000 #0 10 100
            do
                for pos_size in 1
                do
                    for noise in 0.0
                    do
                        for pvaltsp in 0.6 0.8 #0.5 0.6 0.7 0.8 0.9 0.99
                        do
                            for t0_ate in -1.0 -0.7 -0.5 -0.3 -0.1 0.0 0.1 0.3 0.5 0.7 1.0 #-20.0 -10.0 -5.0 -1.0 -0.5 -0.1 0.0 0.1 0.5 1.0 5.0  10.0 20.0 #-0.8 
                            do 
                                python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pvaltsp($pvaltsp).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -sp_topic_pval $pvaltsp -degree_confoundedness $dcf  -topic_name $topic_name -hinge_width $hwidth & #--bert_as_encoder -transformer $transformer --train_bert
                            done
                        done
                    done
                done

                #Waiting for the jobs to complete. Cannot handle so many parallel background jobs
                for job in `jobs -p`
                do
                    wait $job
                done 

            done
        done
    done
done

