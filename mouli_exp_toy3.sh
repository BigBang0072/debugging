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
hlayer=0



# stage_mode="stage2_te_reg_strong_with_width"
stage_mode="stage1_mouli"
debug_tidx=0


#Dataset specific parameters
path="dataset/nlp_toy3/"
out_path="dataset/nlp_toy3"
dtype="toynlp3"
num_topics=1
topic_name="spurious" #causal or spurious --> based on them the dataset will be created
sample=1000
dcf=0.0
pos_size=1



for run_num in 0 1 2 #14 15 #10 11 12 #1 2
do
    for noise in 0.0
    do
        for dcf in 0.0 0.5 1.0
        do
            for pvaltsp in 0.5 0.6 0.7 0.8 0.9 0.95 0.99
            do
                for mvsel_mode in "loss" "acc"
                do 
                    python transformer_debugger.py -expt_num "cad.moulis1toy3.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pvaltsp($pvaltsp).dcf($dcf).mvsel($mvsel_mode)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size  -run_num $run_num -dropout_rate 0.0 -debug_tidx $debug_tidx -stage_mode $stage_mode -sp_topic_pval $pvaltsp -degree_confoundedness $dcf  -topic_name $topic_name -mouli_valid_sel_mode $mvsel_mode &    #--bert_as_encoder -transformer $transformer --train_bert
                done
            done
        done
    done
done

