#Training the bert model
for dset in "twitter_pan16_mention_gender"
do
    for model_type in "bert-base-uncased"
    do
        for neg1_flip_method in "remove_negation"
        do
            for loss_type in "x_entropy"
            do
                for hretrain in "no_warm_encoder"
                do
                    for r in 0
                    do
                        for dropout_rate in 0.0
                        do
                            for l2_lambd in 0.0
                            do
                                for e in 5
                                do
                                    for d in "non_causal"
                                    do
                                        for h in 0
                                        do
                                            for s in 10000
                                            do
                                                for n in 0.0
                                                do
                                                    for p in 0.5 0.6 0.7 0.8 0.9 0.99
                                                    do
                                                        python transformer_debugger.py -expt_num "pt.rel.dset($dset).mt($model_type).neg1_fmethod($neg1_flip_method).lt($loss_type).drate($dropout_rate).l2($l2_lambd).hretrain($hretrain).d($d).n($n).h($h).s($s).e($e).p($p).r($r)" -num_sample $s -num_topics 1 -num_epochs $e -path "dataset/$dset" -emb_path "glove-wiki-gigaword-100" -noise_ratio $n -num_hidden_layer $h -stage 2 -main_model_mode $d --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -neg1_flip_method $neg1_flip_method --measure_flip_pdelta -gpu_num 0 -run_num $r
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
done