import pandas as pd
import pdb
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pprint import pprint

word_actual_neg = ["worst","boring","bad","terrible","awful","not","poorly","dull"]
word_actual_pos = ["awesome","fantastic","well","wonderful","best","good","excellent","great"]
# word_neutral = ["romance","horror","scary","science","tragedy","money"]
word_neutral = ["science","tragedy","money","age","time"]

def read_file(path):
    df = pd.read_csv(path,delimiter=" ",header=None)
    # df = df.apply(str.encode("utf-8"))

    def get_float(str):
        num = float(str[2:-1])
        return num
    
    def get_word(str):
        word = str[2:-1]
        return word

    df[0] = df[0].apply(get_word)
    df[1] = df[1].apply(get_float)
    print(df.head())

    word_imp_dict={}
    for idx in range(df.shape[0]):
        word,imp = df.iloc[idx][0],df.iloc[idx][1]
        word_imp_dict[word]=imp

    return word_imp_dict

def analyze_amazon_experiments(expt_name,num_domains,domain_names):
    imp_dict_list = []
    for dnum in range(num_domains):
        imp_dict = read_file("embeddings/importance_{}.1.{}.tsv".format(expt_name,domain_names[dnum]))
        imp_dict_list.append(imp_dict)
    
    #Now we will plot the variation of certain chosen words
    figure, axes = plt.subplots(3,1)

    #Getting the scores and plotting them
    def get_score_and_plot(word_list,ax,imp_dict_list):
        for word in word_list:
            #Getting the importance of word in each doamin experiemnts
            word_imp = [imp_dict_list[didx][word] for didx in range(num_domains)]

            #Plotting the importance
            ax.plot(word_imp,"o-",label=word)
        ax.legend()
        ax.set_ylim([0,1.02])
        ax.grid(True)

    #Now plotting each of the groups one by one
    get_score_and_plot(word_actual_pos,axes[0],imp_dict_list)
    get_score_and_plot(word_actual_neg,axes[1],imp_dict_list)
    get_score_and_plot(word_neutral,axes[2],imp_dict_list)


     

    plt.show()

    #Plotting the mean and variance of imporrance score
    def get_domain_mean_std(imp_dict):
        dmean = np.mean(list(imp_dict.values()))
        dstd =  np.std(list(imp_dict.values()))

        return dmean,dstd 

    dmean,dstd = zip(*[
        get_domain_mean_std(imp_dict_list[dnum])
                for dnum in range(num_domains)
    ])
    plt.errorbar(list(range(num_domains)),dmean,yerr=dstd,fmt='o-',alpha=0.6,
                            capsize=5,capthick=2,linewidth=2)

    axes = plt.gca()
    axes.set_ylim([-1.02,1.02]) 
    axes.grid(True) 
    plt.show()

    #Plotting the aggregate var dist
    plot_mean_segreagated_var_dist(imp_dict_list)


def plot_mean_segreagated_var_dist(imp_dict_list):
    '''
    Here we will divide the entire importance score section into 4 or 3 parts
    0-0.25, 0.25-0.75, 0.75-1

    and then plot the variance in each of the section.

    from here we can then furthur drill down on varaince score in each of the
    individual region.
    '''
    #Conveting the imp dict list into 
    common_vocab = set.intersection(
        *[set(imp_dict_list[didx]) for didx in range(len(imp_dict_list))]
    )

    #Now aggregating the word importance from each domain in one list
    imp_dict = {word:[
                    imp_dict_list[didx][word] 
                        for didx in range(len(imp_dict_list))
        ] 
        for word in common_vocab
    }

    #Now getting the mean and standard deviation for each of the words
    imp_agg_list = [
                (word,np.mean(imp_dict[word]),np.std(imp_dict[word]))
                        for word in imp_dict.keys()
    ]


    #Now plotting the variance distribution in each of the section
    def plot_bucket_metrics(imp_agg_list,bucket_uplims):
        #First of all we segment the whole list into these segments
        bucket_dict = defaultdict(list)
        for agg_info in imp_agg_list:
            #Putting this words and its info in bucket
            for bidx,buplim in enumerate(bucket_uplims):
                if(agg_info[1]<=buplim):
                    bucket_dict[bidx].append(agg_info)
                    break
        
        #Now we will fill/plot each of the bucket one by one
        figure, axes = plt.subplots(len(bucket_uplims),1)

        def fill_the_bucket(bucket_list,bidx,buplim,ax):
            print("#############################################")
            print("#############################################")
            print("Filling the bucket:{}\tuplim:{}".format(bidx,buplim))

            #First of all sorting this bucket based on the variance
            bucket_list.sort(key=lambda x:x[-1])

            #printing the bottom and top few:
            print("Bottom words on bucket: (based on variance)")
            pprint(bucket_list[0:20])
            print("Top words in this bucket:")
            pprint(bucket_list[-20:])

            #Now filling the plot
            std_list = [agg_info[-1] for agg_info in bucket_list]
            ax.hist(std_list,ec="k")
            ax.set_xlim([-0.1,1.1])
            ax.grid(True)
            ax.set_title("Mean Importance <= :{}".format(buplim))
        
        #Filling the individual buckets
        for bidx,buplim in enumerate(bucket_uplims):
            fill_the_bucket(bucket_dict[bidx],bidx,buplim,axes[bidx])
        
        plt.show()
    
    bucket_uplims = [0.5,0.75,0.9,1.0]
    plot_bucket_metrics(imp_agg_list,bucket_uplims)

    pdb.set_trace()



    
def analyze_imdb_experiments(expt_name):
    imp1_dict = read_file("embeddings/importance_{}.single.tsv".format(expt_name))
    imp2_dict = read_file("embeddings/importance_{}.both.tsv".format(expt_name))

    word1_list = set(imp1_dict.keys())
    word2_list = set(imp2_dict.keys())
    print("list1:",len(word1_list))
    print("list2:",len(word2_list))

    common_words = word2_list.intersection(word1_list)
    print("Num Common:",len(common_words))

    #Gettiing the difference
    diff_list = []
    diff_dict = {}
    for word in common_words:
        diff = imp2_dict[word] - imp1_dict[word]
        diff_list.append((word,diff))
        diff_dict[word]=diff
    
    diff_list.sort(key=lambda x:x[1])



    

    def print_diff(word_list):
        diff=["{}\t\t{:0.3f}\t\t{:0.3f}".format(word,imp1_dict[word],diff_dict[word]) for word in word_list]
        print("\n".join(diff))
    

    print("#######################################")
    print_diff(word_actual_neg)
    print("#######################################\n\n")
    print_diff(word_actual_pos)
    print("#######################################\n\n")
    print_diff(word_neutral)
    print("#######################################\n\n")

if __name__=="__main__":
    expt_name = "18.amzn"
    num_domains = 8
    domain_names = ["arts","books","phones","clothes","groceries","movies","pets","tools"]

    analyze_amazon_experiments(expt_name,num_domains,domain_names)




    pdb.set_trace()


