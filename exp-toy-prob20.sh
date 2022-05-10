# Training the BERT model for the adversarial removal
for model_type in "nbow"
do
    for neg1_flip_method in "remove_negation"
    do
        for loss_type in "x_entropy"
        do
            for hretrain in "no_warm_encoder"
            do
                for run_num in 1
                do
                    for noise in 0.0
                    do
                        for sample in 1000
                        do
                            for l2_lambd in 0.0
                            do
                                for dropout_rate in 0.0
                                do
                                    for mainepoch in 20
                                    do
                                        for mainmode in "causal_removed_sp"
                                        do
                                            for hlayer in 0
                                            do
                                                for advepoch in 20
                                                do
                                                    for remmode in "probing"
                                                    do
                                                        for pval in 0.99 0.9 0.8 0.7 0.6 0.5
                                                        do
                                                            python transformer_debugger.py -expt_num "pt.rel.mt($model_type).neg1_fmethod($neg1_flip_method).remmode($remmode).adv_rm_method($adv_rm_method).advepoch($advepoch).lt($loss_type).drate($dropout_rate).l2($l2_lambd).n($noise).h($hlayer).s($sample).e($mainepoch).p($pval).r($run_num)" -num_sample $sample -num_topics 2 -num_epochs $mainepoch -path "dataset/nlp_toy2/"  -emb_path "glove-wiki-gigaword-100" -noise_ratio $noise -num_hidden_layer $hlayer -stage 2 -main_model_mode $mainmode --normalize_emb -lr 5e-3 -head_retrain_mode $hretrain -l2_lambd $l2_lambd -loss_type $loss_type -dropout_rate $dropout_rate  -adv_rm_epochs $advepoch -debug_tidx 1 -removal_mode $remmode -dtype "toynlp2"  -topic0_corr 1.0 -topic1_corr $pval  -neg1_flip_method $neg1_flip_method --measure_flip_pdelta -run_num $run_num --valid_before_gupdate
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