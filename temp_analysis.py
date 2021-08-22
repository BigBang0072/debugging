import pandas as pd
import pdb

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


if __name__=="__main__":
    expt_name = "4"
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
    
    word_actual_neg = ["worst","boring","bad","terrible","awful","not","poorly","dull"]
    word_actual_pos = ["awesome","fantastic","well","wonderful","best","good","excellent","great"]
    word_neutral = ["romance","horror","scary","comedy","science","scifi","tragedy","disco"]

    print("#######################################")
    print_diff(word_actual_neg)
    print("#######################################\n\n")
    print_diff(word_actual_pos)
    print("#######################################\n\n")
    print_diff(word_neutral)
    print("#######################################\n\n")




    pdb.set_trace()


