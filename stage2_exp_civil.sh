



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
path="dataset/civilcomments/"
out_path="dataset/civilcomments"
dtype="civilcomments"
num_topics=1
replace_strategy="remove" #"map_replace" 
sample=5000
topic_name="race"



#13,14,15 with increased probabilty 0.99,0.30,0.70,0.01
#10,11,12 with previou probabiltiy 0.9,0.4,0.6,0.1
for run_num in 0 #1 2 
do
    for dcf in 0.0
    do 
        for hwidth in 0.0 #DONT not registered as fname #0.0 #0.05 0.1 0.2 0.5 1.0
        do 
            for te_lambda in 100 #0 10 100 1000 4000 10000 #10 100 1000 4000 10000
            do 
                for pos_size in 1
                do
                    for noise in 0.0
                    do
                        for pval in 0.9 #0.5 0.6 0.7 0.8 0.9 0.99
                        do
                            for t0_ate in 0.0
                            do 
                                python transformer_debugger.py -expt_num "cad.civils2debug.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pvaltsp($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv"  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_pval $pval -degree_confoundedness $dcf  -topic_name $topic_name -hinge_width $hwidth --bert_as_encoder -transformer $transformer --train_bert -replace_strategy $replace_strategy -max_len $max_len
                            done
                        done
                    done
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


