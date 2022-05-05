#Training the bert model
for model_type in "roberta-base"
do
    for neg1_flip_method in "remove_negation"
    do
        for loss_type in "x_entropy"
        do
            for hretrain in "no_warm_encoder"
            do
                for r in 1 2 3
                do
                    for dropout_rate in 0.0
                    do
                        for l2_lambd in 0.0
                        do
                            for e in 20
                            do
                                for d in "non_causal"
                                do
                                    for h in 0
                                    do
                                        for s in 10000
                                        do
                                            for n in 0.0
                                            do
                                                for p in 0.99 0.9 0.8 0.7 0.6 0.5
                                                do
                                                    python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).neg1_fmethod($neg1_flip_method).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 1 -num_epochs $e -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -neg1_flip_method $neg1_flip_method --measure_flip_pdelta -gpu_num 0 -run_num $r
                                                    git add .
                                                    git commit -m "BOT:Saving the result for convergence large epoch"
                                                    git push origin nlpct:nlpct_conve20_$neg1_flip_method
                                                done
                                            done
                                        done
                                    done       
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done