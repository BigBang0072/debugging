#Training the null-space model for the BERT
for model_type in "bert-base-uncased"
do
    for neg1_flip_method in "replace_negation"
    do
        for loss_type in "x_entropy"
        do
            for hretrain in "no_warm_encoder"
            do
                for run in 0
                do
                    for noise in 0.0
                    do
                        for sample in 10000
                        do
                            for hANDl2 in 0,0.0
                            do
                                IFS=',' read hlayer l2_lambd <<< "${hANDl2}"
                                for mainepoch in 3
                                do
                                    for mainmode in "causal"
                                    do
                                        for advepoch in 20
                                        do
                                            for topicepoch in 1
                                            do
                                                for remmode in "null_space"
                                                do
                                                    for p in 0.99 0.9 0.8 0.7 0.6 0.5
                                                    do
                                                        python transformer_debugger.py -expt_num "pt.rel.model_type($model_type).neg1_fmethod($neg1_flip_method).remmode($remmode).advepoch($advepoch).topicepoch($topicepoch).lt($loss_type).l2($l2_lambd).hretrain($hretrain).mainmode($mainmode).noise($noise).hlayer($hlayer).sample($sample).mainepoch($mainepoch).p($p).run($run)" -num_sample $sample -num_topics 1 -num_epochs $mainepoch -path "dataset/multinli_1.0/" -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-5 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -num_proj_iter $advepoch  -debug_tidx 0 -removal_mode $remmode -topic_epochs $topicepoch -dtype "nlp_bert" --bert_as_encoder --train_bert -neg_topic_corr $p -transformer $model_type -neg1_flip_method $neg1_flip_method --measure_flip_pdelta -gpu_num 2
                                                        git add .
                                                        git commit -m "BOT:Saving the result for inlp removal large epoch"
                                                        git push origin nlpct:nlpct_inlpe20
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