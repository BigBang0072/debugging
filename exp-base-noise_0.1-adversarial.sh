for model_type in "bert-base-uncased"
do
    for loss_type in "x_entropy"
    do
        for hretrain in "no_warm_encoder"
        do
            for run in 0
            do
                for noise in 0.1
                do
                    for sample in 10000
                    do
                        for l2_lambd in 0.0
                        do
                            for dropout_rate in 0.0
                            do
                                for mainepoch in 0
                                do
                                    for mainmode in "non_causal"
                                    do
                                        for hlayer in 0
                                        do
                                            for advepoch in 5
                                            do
                                                for grstrength in 0.01 0.1 0.5 1.0 2.0 4.0 8.0
                                                do
                                                    for remmode in "adversarial"
                                                    do
                                                        for adv_rm_method in "adv_rm_with_main"
                                                        do
                                                            for p in 0.99 0.9 0.8 0.7 0.6 0.5
                                                            do
                                                                python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).remmode($remmode).adv_rm_method($adv_rm_method).grstrength($grstrength).advepoch($advepoch).lt($loss_type).dropout_rate($dropout_rate).l2($l2_lambd).hretrain($hretrain).noise($noise).hlayer($hlayer).sample($sample).p($p).run($run)" -num_sample $sample -num_topics 1 -num_epochs $mainepoch -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -adv_rm_epochs $advepoch -rev_grad_strength $grstrength -debug_tidx 0 -removal_mode $remmode -dropout_rate $dropout_rate -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -adv_rm_method $adv_rm_method -neg1_flip_method "dont_measure" --measure_flip_pdelta
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
    done
done