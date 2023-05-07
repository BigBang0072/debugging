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



# stage_mode="stage2_te_reg_strong_with_width"
stage_mode="stage1_mouli_cad" # stage1_mouli_te_reg_strong     stage1_mouli_cad
debug_tidx=0


#Dataset specific parameters
path="dataset/mnist/"
out_path="dataset/mnist"
dtype="mnist"
num_topics=1
# topic_name="" #causal or spurious --> based on them the dataset will be created
sample=10000
dcf=0.0
pos_size=1



for run_num in 0  #1 2 #1 2 #1 2 #14 15 #10 11 12 #1 2
do
    for topic_name in "all" # for mouli expt we internally give topic name
    do 
        for noise in 0.3
        do
            for hlayer in 1 
            do 
                for dcf in 0.0 #0.5 1.0
                do
                    for te_lambda in  0 #1 10 100 1000 10000
                    do 
                        for pvaltsp in 0.5 #0.5 0.6 0.7 0.8 0.9 0.95 0.99
                        do
                            for mvsel_mode in "loss" #"acc"
                            do 
                                python transformer_debugger.py -expt_num "cad.moulis1mnist.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pvaltsp($pvaltsp).dcf($dcf).mvsel($mvsel_mode).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx -stage_mode $stage_mode -sp_topic_pval $pvaltsp -degree_confoundedness $dcf  -topic_name $topic_name -mouli_valid_sel_mode $mvsel_mode  -te_lambda $te_lambda  & #--bert_as_encoder -transformer $transformer --train_bert
                            done
                        done
                    done 
                done
            done
        done
    done 
    #Pausing for the te lambads
    # for job in `jobs -p`
    # do
    #     wait $job
    # done
done

