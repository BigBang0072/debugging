from distutils.command import clean
import numpy as np
import json
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict


#Calculating the spuriousness score
def calculate_spuriousness_score(curr_smin,clean_smin):
    '''
    '''
    return np.abs(1.0 - (curr_smin*1.0)/(clean_smin*1.0))

def calculate_spuriousness_score_estimate(smin_acc_list,clean_smin_acc_list):
    '''
    Given list of multiple runs of current classifier and the 
    multiple runs of clean classifier --> esrimate the clean acc list 

    calculte the spuriousness scrore
    '''
    sp_score_estm_list=[]
    for curr_smin in smin_acc_list:
        for clean_smin in clean_smin_acc_list:
            sp_score_estm_list.append(
                        calculate_spuriousness_score(
                                    curr_smin=curr_smin,
                                    clean_smin=clean_smin
                        )
            )
    
    mean_sp_score = np.mean(sp_score_estm_list)
    std_sp_score = np.std(sp_score_estm_list)
    return sp_score_estm_list

#Opening the experiment json
def load_probe_metric_list(fname,only_one=False,epoch=None):
    with open(fname,"r") as rhandle:
        probe_metric_list = json.load(rhandle)
    
    #Converting the metric into usable format
    pdict = defaultdict(list)
    
    #Getting only the final outcome if only_last required
    if only_one==True:
        probe_metric_list = probe_metric_list[epoch:epoch+1]#[-10:-9]
    
    for idx in range(len(probe_metric_list)):
#         pdict["angle:m-t0"].append(probe_metric_list[idx]["conv_angle_dict"]["main"]["topic0"])
        pdict["acc:main"].append(probe_metric_list[idx]["classifier_acc_dict"]["main"])
        if "pp_emb_norm" in probe_metric_list[idx]["classifier_acc_dict"]:
            pdict["pp_emb_norm"].append(probe_metric_list[idx]["classifier_acc_dict"]["pp_emb_norm"])
        if "emb_norm" in probe_metric_list[idx]["classifier_acc_dict"]:
            pdict["emb_norm"].append(probe_metric_list[idx]["classifier_acc_dict"]["emb_norm"])
        

        pdict["acc:topic0"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0"])
        pdict["topic0_main"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main"])
        pdict["topic0_pdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta"])
        
        pdict["topic0_logpdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_logpdelta"])
        pdict["topic0_pdelta_m1t0"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta_m1t0"])
        pdict["topic0_pdelta_m0t0"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta_m0t0"])
        pdict["topic0_pdelta_m1t1"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta_m1t1"])
        pdict["topic0_pdelta_m0t1"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta_m0t1"])
        pdict["topic0_pdelta_all"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta_all"])
        pdict["topic0_pdelta_smin"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_pdelta_smin"])

        pdict["main_smin_topic0"].append(probe_metric_list[idx]["classifier_acc_dict"]["main_smin_topic0"])
        pdict["topic0_smin"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_smin_main"])
        
        
        if "topic0_pos_con_loss" in probe_metric_list[idx]["classifier_acc_dict"]:
            pdict["topic0_pos_con_loss"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_pos_con_loss"])
            pdict["topic0_neg_con_loss"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_neg_con_loss"])
            pdict["topic0_last_emb_norm"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_last_emb_norm"])


        if "topic1" in probe_metric_list[idx]["classifier_acc_dict"]:
#             pdict["angle:m-t1"].append(probe_metric_list[idx]["conv_angle_dict"]["main"]["topic1"])
#             pdict["angle:t0-t1"].append(probe_metric_list[idx]["conv_angle_dict"]["topic0"]["topic1"])

            pdict["acc:topic1"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1"])
            pdict["topic1_main"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main"])
            pdict["topic1_pdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta"])

            
            pdict["main_smin_topic1"].append(probe_metric_list[idx]["classifier_acc_dict"]["main_smin_topic1"])
            pdict["topic1_smin"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_smin_main"])

            pdict["topic1_logpdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_logpdelta"])
            pdict["topic1_pdelta_m1t0"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta_m1t0"])
            pdict["topic1_pdelta_m0t0"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta_m0t0"])
            pdict["topic1_pdelta_m1t1"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta_m1t1"])
            pdict["topic1_pdelta_m0t1"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta_m0t1"])
            pdict["topic1_pdelta_all"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta_all"])
            pdict["topic1_pdelta_smin"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta_smin"])

        # pdict["acc:topic1"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1"])
        # pdict["topic1_main"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main"])
        # pdict["topic1_pdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_pdelta"])
        if "topic1_flip_emb_diff" in probe_metric_list[idx]["classifier_acc_dict"]:
            pdict["topic1_emb_diff"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_emb_diff"])
            pdict["topic0_emb_diff"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_emb_diff"])
        if "topic1_flip_main_logpdelta" in probe_metric_list[idx]["classifier_acc_dict"]:
            pdict["topic0_logpdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic0_flip_main_logpdelta"])
            pdict["topic1_logpdelta"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_flip_main_logpdelta"])
            
        if "topic1_pos_con_loss" in probe_metric_list[idx]["classifier_acc_dict"]:
            pdict["topic1_pos_con_loss"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_pos_con_loss"])
            pdict["topic1_neg_con_loss"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_neg_con_loss"])
            pdict["topic1_last_emb_norm"].append(probe_metric_list[idx]["classifier_acc_dict"]["topic1_last_emb_norm"])
        
    return pdict

#Collecting multiple runs of experiment
def aggregate_random_runs(rdict_list):
    '''
    '''
    #This is done assuming we have only one value per rdict i.e after convergence.
    rdict_agg_list = defaultdict(list)
    for rdict in rdict_list:
        for key,val in rdict.items():
            rdict_agg_list[key]+=val
    
    #Now taking the mean of the experiments
    rdict_agg = defaultdict(list)
    for key,val in rdict_agg_list.items():
        rdict_agg[key] = dict(
                            mean = np.mean(val),
                            std  = np.std(val),
                            val_list = val,
        )
    
    return rdict_agg

def aggregate_random_runs_timeline(prdict_list):
    '''
    '''
    rdict_agg = defaultdict(dict)
    #Getting the aggregate metric for each of the key
    for key in prdict_list[0].keys():
        #Getting the aggregate for this key
        timeline_run_list = []
        for ridx in range(len(prdict_list)):
            timeline_run_list.append(prdict_list[ridx][key])
        timeline_run_matrix = np.array(timeline_run_list)
        #Getting the aggreget acorss axis =0 [[timeline_run1],[timeline_run2]...]
        rdict_agg[key]["mean"] = np.mean(timeline_run_matrix,axis=0)
        rdict_agg[key]["std"]  = np.std(timeline_run_matrix,axis=0)
        rdict_agg[key]["val_mat"] = timeline_run_matrix
    return rdict_agg

def get_all_result_dict(run_list,pval_list,enum,fname_pattern,clean_smin_acc_list_dict=None):
    all_result_dict = {}
    for pidx,pval in enumerate(pval_list):
        prdict_list = []
        for nidx in run_list:
            fname = fname_pattern.format(pval,nidx)
            prdict = load_probe_metric_list(
                                fname,
                                only_one=True,
                                epoch=enum-1,
            )
            prdict_list.append(prdict)
        #Getting the aggregate result
        prdict_agg = aggregate_random_runs(prdict_list)

        #Now claculating the spuriousness score for the desrired classifiers
        if clean_smin_acc_list_dict!=None:
            for metric_name in clean_smin_acc_list_dict.keys():
                smin_acc_list = prdict_agg[metric_name]["val_list"]
                clean_smin_acc_list = clean_smin_acc_list_dict[metric_name]
                sp_score_estm_list = calculate_spuriousness_score_estimate(
                                        smin_acc_list=smin_acc_list,
                                        clean_smin_acc_list=clean_smin_acc_list
                )
                #Adding the result to the the aggregate dict
                prdict_agg[metric_name+"_sp_score"]=dict(
                                            mean=np.mean(sp_score_estm_list),
                                            std=np.std(sp_score_estm_list),
                                            val_list=sp_score_estm_list,
                )
        #Adding the aggregate dict to al result dict
        all_result_dict[pval] = prdict_agg

    return all_result_dict

def get_all_result_timeline(run_list,pval_list,fname_pattern,clean_smin_acc_list_dict=None):
    all_result_timeline={}
    for pval in pval_list:
        prdict_list = []
        for ridx in run_list:
            fname = fname_pattern.format(pval,ridx)
            # print("Loading file: {}".format(fname))
            prdict = load_probe_metric_list(fname)
            prdict_list.append(prdict)
        #Getting the aggregate list
        prdict_agg = aggregate_random_runs_timeline(prdict_list)
        #Getting the degree of spuriousness score for every timestep
        if clean_smin_acc_list_dict!=None:
            for metric_name in clean_smin_acc_list_dict.keys():
                #Now going one time step at a time and getting the score
                timestep_mean_sp_score_list=[]
                # print(prdict_agg)
                for tidx in range(prdict_agg[metric_name]["mean"].shape[0]):
                    smin_acc_list = prdict_agg[metric_name]["val_mat"][:,tidx]
                    clean_smin_acc_list = clean_smin_acc_list_dict[metric_name]
                    sp_score_estm_list = calculate_spuriousness_score_estimate(
                                        smin_acc_list=smin_acc_list,
                                        clean_smin_acc_list=clean_smin_acc_list
                    )
                    timestep_mean_sp_score_list.append(sp_score_estm_list)
                #Creating the sp score matrix
                timestep_mean_sp_score_mat = np.array(timestep_mean_sp_score_list).T
                #Now saving the aggregate
                prdict_agg[metric_name+"_sp_score"]=dict(
                                            mean=np.mean(timestep_mean_sp_score_mat,axis=0),
                                            std=np.std(timestep_mean_sp_score_mat,axis=0),
                                            val_mat=timestep_mean_sp_score_mat,
                )


        #Saving the aggregate list
        all_result_timeline[pval] = prdict_agg
    return all_result_timeline

def plot_all_results(ax,pval_list,all_result_dict,plot_item_list,plot_item_custname=None,extra_label=""):
    '''
    '''
    #Now we can reuse the previous ax
    if(type(ax)!=type(np.array([1,2]))):
        fig,ax = plt.subplots(len(plot_item_list),)
        
    for iidx,item_name in enumerate(plot_item_list):
        #Colletcing the metrics
        yval = [all_result_dict[pval][item_name]["mean"] for pval in pval_list]
        yerr = [all_result_dict[pval][item_name]["std"] for pval in pval_list]
        ylb = [
            all_result_dict[pval][item_name]["mean"]-all_result_dict[pval][item_name]["std"]
                for pval in pval_list
        ]
        yub = [ 
            all_result_dict[pval][item_name]["mean"]+all_result_dict[pval][item_name]["std"]
                for pval in pval_list
        ]
        #Plotting the guy
        ax[iidx].errorbar(pval_list,yval,yerr,ls="-.",marker="o",label=extra_label,alpha=0.7)
        # ax[iidx].fill_between(pval_list,ylb,yub,alpha=0.2)


        #Setting the plot attributes
        # ax[iidx].set_ylim(0.0,1.0)
        ax[iidx].set_xlabel("predictive-correlation between invariant and spurious attribute")

        if plot_item_custname!=None:
            ylabel = plot_item_custname[iidx]
        else:
            ylabel = item_name
        ax[iidx].set_ylabel(ylabel)



        ax[iidx].legend()
        ax[iidx].grid(True)
        ax[iidx].set_title(item_name)
    return ax

def plot_one_timeline(pval_list,all_result_timeline,plot_item_list):
    fig,ax = plt.subplots(len(plot_item_list),)
    
    for pval in pval_list:
        pdict_timeline = all_result_timeline[pval]
        for iidx,item_name in enumerate(plot_item_list):
            x_val = range(len(pdict_timeline[item_name]["mean"])) #Denotes the number of epoch
            y_val = pdict_timeline[item_name]["mean"]
            y_err = pdict_timeline[item_name]["std"]
            ax[iidx].errorbar(x_val,y_val,y_err,marker="o",ls="-.",label="pval={}".format(pval),alpha=0.7)
            ax[iidx].set_xlabel("epochs")
            ax[iidx].set_ylabel(item_name)
            ax[iidx].legend()
            ax[iidx].grid(True)
            ax[iidx].set_title(item_name)
            ax[iidx].set_xticks(x_val)
    plt.show()