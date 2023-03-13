



#STAGE2-METHOD2 EXPT
mainepoch=5
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



#Regularizing with true effect and then with DR
# for topicANDsample in "gender",5000
# do 
#     IFS=',' read topic_name sample <<< "${topicANDsample}"
#     for run_num in 0 1 2
#     do
#         for te_lambda in  1
#         do
#             for pos_size in 1
#             do
#                 for noise in 0.0
#                 do
#                     for pvalANDt0_ate in 0.5,0.05 0.6,0.02 0.7,0.05 0.8,0.05 0.9,0.15 0.99,0.1 
#                     do
#                         IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

#                         python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert 
#                     done
#                 done
#             done
#         done
#     done
# done

# for topicANDsample in "gender",5000
# do 
#     IFS=',' read topic_name sample <<< "${topicANDsample}"
#     for run_num in 1 2
#     do
#         for te_lambda in  1
#         do
#             for pos_size in 1
#             do
#                 for noise in 0.0
#                 do
#                     for pvalANDt0_ate in   0.5,0.05 0.6,0.04 0.7,0.04 0.8,0.05 0.9,0.11 0.99,0.15
#                     do
#                         IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

#                         python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert 
#                     done
#                 done
#             done
#         done
#     done
# done









# for topicANDsample in "race",5000
# do 
#     IFS=',' read topic_name sample <<< "${topicANDsample}"
#     for run_num in 1 2
#     do
#         for te_lambda in  1
#         do
#             for pos_size in 1
#             do
#                 for noise in 0.0
#                 do
#                     for pvalANDt0_ate in  0.5,-0.03 0.6,0.06 0.7,0.1 0.8,0.2 0.9,0.18 0.99,0.21   0.5,-0.01 0.6,0.06 0.7,0.12 0.8,0.05 0.9,0.3 0.99,0.24
#                     do
#                         IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

#                         python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert 
#                     done
#                 done
#             done
#         done
#     done
# done




# for topicANDsample in "religion",4000
# do 
#     IFS=',' read topic_name sample <<< "${topicANDsample}"
#     for run_num in 1 2
#     do
#         for te_lambda in  1
#         do
#             for pos_size in 1
#             do
#                 for noise in 0.0
#                 do
#                     for pvalANDt0_ate in  0.5,0.0 0.6,0.0 0.7,0.08 0.8,0.05 0.9,0.08 0.99,0.1   0.5,0.01 0.6,0.02 0.7,0.04 0.8,0.09 0.9,0.09 0.99,0.18
#                     do
#                         IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

#                         python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert 
#                     done
#                 done
#             done
#         done
#     done
# done






#=================================================================
#Running the random TE experiemnt
#=================================================================
# for topicANDsample in "religion",4000
# do 
#     IFS=',' read topic_name sample <<< "${topicANDsample}"
#     for run_num in  1 2
#     do
#         for te_lambda in  1
#         do
#             for pos_size in 1
#             do
#                 for noise in 0.0
#                 do
#                     for pvalANDt0_ate in  0.5,0.66 0.6,0.46 0.7,0.13 0.8,0.67 0.9,0.25 0.99,0.24  0.5,0.20 0.6,0.43 0.7,0.18 0.8,0.07 0.9,0.30 0.99,0.27
#                     do
#                         IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

#                         python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -replace_strategy $replace_strategy 
#                     done
#                 done
#             done
#         done
#     done
# done


for topicANDsample in "race",5000
do 
    IFS=',' read topic_name sample <<< "${topicANDsample}"
    for run_num in 0 1 2
    do
        for te_lambda in  1
        do
            for pos_size in 1
            do
                for noise in 0.0
                do
                    for pvalANDt0_ate in  0.5,0.66 0.6,0.46 0.7,0.13 0.8,0.67 0.9,0.25 0.99,0.24  0.5,0.20 0.6,0.43 0.7,0.18 0.8,0.07 0.9,0.30 0.99,0.27
                    do
                        IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

                        python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -replace_strategy $replace_strategy
                    done
                done
            done
        done
    done
done



for topicANDsample in "gender",5000
do 
    IFS=',' read topic_name sample <<< "${topicANDsample}"
    for run_num in 0 1 2
    do
        for te_lambda in  1
        do
            for pos_size in 1
            do
                for noise in 0.0
                do
                    for pvalANDt0_ate in  0.5,0.66 0.6,0.46 0.7,0.13 0.8,0.67 0.9,0.25 0.99,0.24  0.5,0.20 0.6,0.43 0.7,0.18 0.8,0.07 0.9,0.30 0.99,0.27
                    do
                        IFS=',' read pval t0_ate <<< "${pvalANDt0_ate}"

                        python transformer_debugger.py -expt_num "cad.civils2.rnum($run_num).topic($topic_name).sample($sample).noise($noise).pval($pval).t0_ate($t0_ate).telambda($te_lambda)" -num_sample $sample -num_topics $num_topics -num_epochs $mainepoch -cfactuals_bsize $cbsize -path $path  -out_path $out_path -emb_path "glove-wiki-gigaword-100" -vocab_path "assets/word2vec_10000_200d_labels.tsv" -max_len $max_len  -noise_ratio $noise -num_hidden_layer $hlayer  -main_model_mode $main_mode --normalize_emb -lr $lr -batch_size $batch_size -dtype $dtype -loss_type "x_entropy" -teloss_type $teloss_type -cfactuals_bsize $cbsize  -num_pos_sample $pos_size -num_neg_sample $pos_size -te_lambda $te_lambda -run_num $run_num -dropout_rate 0.0 -t0_ate $t0_ate -debug_tidx $debug_tidx -stage_mode $stage_mode -topic_name $topic_name -topic_pval $pval --bert_as_encoder -transformer $transformer --train_bert -replace_strategy $replace_strategy
                    done
                done
            done
        done
    done
done