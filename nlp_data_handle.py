import numpy as np
import pandas as pd 
import tensorflow as tf
import gensim.downloader as gensim_api

import gzip
import json
import re 
import pprint
import pdb
pp=pprint.PrettyPrinter(indent=4)

class DataHandler():
    '''
    This class will handle all the dataset we will use in our nlp experiemnts
    '''
    dict_w2i = None         #Mapping word to index in embedding space
    dict_i2w = None         #Mapping the index to word
    vocab_len = None        #The number of words in the vocabulary

    emb_matrix = None       #Embedding matrix

    train_data = None       #List of (label,doc)
    valid_data = None       
    test_data = None

    sample_weight = None        

    def __init__(self,data_args):
        '''
        '''
        self.data_args = data_args

        #Initializing the word2index dict (unknown word at first)
        self.vocab_len = 0
        self._add_word_to_vocab("unk")
        self._add_word_to_vocab("<pad>")

        #Initializing the delimiters
        self.delimiter=",|\?|\!|-|\*| |  |;|\.|\(|\)|\n|\"|:|'|/|&|`|[|]|\{|\}|\>|\<"
   
    def load_embedding_mat(self,):
        '''
        '''
        #We will initialize the embedding matrix randomly and learn on go
        if self.data_args["emb_path"]=="random":
            self.emb_matrix = np.random.randn(len(self.dict_w2i),self.data_args["emb_dim"])
            self.emb_matrix[0,:]=0
            self.emb_matrix[1,:]=0
        else:
            #Loading the embedding from the gensim repo
            print("Loading the WordVectors via Gensim! Hold Tight!")
            emb_model = gensim_api.load(self.data_args["emb_path"])

            #Getting the unk vector to be used in place when nothing is there
            unk_vektor = emb_model.get_vector("unk")

            #Now filtering out the required embeddings
            emb_matrix = []
            for widx in range(self.vocab_len):
                word = self.dict_i2w[widx]
                try:
                    word_vec = emb_model.get_vector(word)
                    emb_matrix.append(word_vec)
                except:
                    print("Missed Word: ",word)
                    emb_matrix.append(unk_vektor)
            #Now we will assign this matrix to the class
            self.emb_matrix = np.stack(emb_matrix,axis=0)

            #Updating the embeddig dimension
            self.data_args["emb_dim"]=self.emb_matrix.shape[-1]
    
    def _add_word_to_vocab(self,word):
        '''
        '''
        if self.dict_w2i==None:
            self.dict_w2i={}
            self.dict_i2w={}
        #Adding the word to vocablury
        self.dict_w2i[word] = self.vocab_len
        self.dict_i2w[self.vocab_len] = word

        #Increasing the size of the vocablury
        self.vocab_len+=1

    def _clean_the_document(self,document):
        #Splitting the document by the delimiters
        tokens = re.split(self.delimiter,document)
        
        #Updating the vocablury dictionary
        doc2idx = []
        for token in tokens:
            if(len(token)==0):
                continue
            token = token.lower()

            #Adding the token to vocab
            if token not in self.dict_w2i:
                self._add_word_to_vocab(token)
            
            #Adding the token idx if not approached max len
            if self.data_args["max_len"]==len(doc2idx):
                break
            doc2idx.append(self.dict_w2i[token])
        
        #Now we will pad the rest of the length
        doc2idx = doc2idx + [self.dict_w2i["<pad>"]]*(self.data_args["max_len"]-len(doc2idx))
        

        #TODO: Later we could remove some of the words --> unk based on frequency
        return doc2idx,len(tokens)

    def data_handler_ltdiff_paper_sentiment(self,path):
        '''
        This function will extract the dataset and get the vocablury for use.
        '''
        #Reading the dataframe
        train_df = pd.read_csv(path+"train.tsv",sep="\t")
        valid_df = pd.read_csv(path+"dev.tsv",sep="\t")
        
        def parse_dataset(df):
            print("\n\n#####################################")
            print("Parsing the data frame:")
            print(df.head())
            print("#####################################")
            data_list = []
            doclens = []
            for eidx in range(df.shape[0]):
                label = 1
                if df.iloc[eidx]["Sentiment"]=="Negative":
                    label = 0
                
                #Now getting the cleaned document
                doc2idx,doclen = self._clean_the_document(df.iloc[eidx]["Text"])

                data_list.append((label,doc2idx))
                doclens.append(doclen)
            
            print("Avg Document Lengths: ", np.mean(doclens))
            print("####################################")
            return data_list
        
        #Now going one by one to the reviews we will create the 
        self.train_data = parse_dataset(train_df)
        self.valid_data = parse_dataset(valid_df)

        return self.train_data,self.valid_data
    
    def data_handler_amazon_reviews(self,path,cat_list,num_sample):
        '''
        cat_list    : list of categories/domain we want to include
        num_sample  : the number of samples from each domain
        '''
        #Parser for the zipped file
        def parse(path):
            g = gzip.open(path,"rb")
            for l in g:
                yield json.loads(l)

        def get_category_df(path,cat,num_sample):
            pos_list = []
            neg_list = []
            pos_doclen = []
            neg_doclen = []
            skip_count = 0
            path = "{}{}.json.gz".format(path,cat)
            for d in parse(path):
                # pdb.set_trace()
                #first of all see if this is relavant (>3: pos ; <3:neg)
                if (d["overall"]==3):
                    continue
                elif (len(pos_list)>=num_sample and len(neg_list)>=num_sample):
                    break
                
                #Now cleaning the text
                if("reviewText" not in d):
                    skip_count  +=1
                    continue 
                doc2idx,doclen = self._clean_the_document(d["reviewText"])
                sentiment = 1 if d["overall"]>3 else 0
                
                if(len(neg_list)<num_sample and sentiment==0):
                    neg_list.append((sentiment,doc2idx))
                    neg_doclen.append(doclen)
                elif (len(pos_list)<num_sample and sentiment==1):
                    pos_list.append((sentiment,doc2idx))
                    pos_doclen.append(doclen)
                
            
            #Getting the stats from this category
            print("cat:{}\tpos:{}\tneg:{}\tpos_avlen:{}\tneg_avlen:{}\tskips:{}".format(
                                        cat,
                                        len(pos_list),
                                        len(neg_list),
                                        np.mean(pos_doclen),
                                        np.mean(neg_doclen),
                                        skip_count,
            ))
            
            #Getting the train and validation dataset
            train_pos_list = pos_list[0:int(self.data_args["train_split"]*len(pos_list))]
            valid_pos_list = pos_list[int(self.data_args["train_split"]*len(pos_list)):]

            train_neg_list = neg_list[0:int(self.data_args["train_split"]*len(neg_list))]
            valid_neg_list = neg_list[int(self.data_args["train_split"]*len(neg_list)):]

            train_list = train_pos_list + train_neg_list 
            valid_list = valid_pos_list + valid_neg_list

            return train_list, valid_list
        
        #Getting the dataframe for each categories
        self.train_data=[]
        self.valid_data=[]
        cat_wise_ssize = []
        for cat in cat_list:
            cat_train_list,cat_valid_list  = get_category_df(path,cat,num_sample)
            
            #Adding the data to the appropriate list
            self.train_data += cat_train_list
            self.valid_data += cat_valid_list

            #Number of samples in each category for weight
            cat_wise_ssize.append(len(cat_train_list))
        
        print("Total Number of training examples:",len(self.train_data))
        print("Total number of validation examples:",len(self.valid_data))

        #Now we will caulcalate the weights for each sample
        total_examples = len(self.train_data)*1.0
        sample_weight = [[cat_ssize/total_examples]*cat_ssize for cat_ssize in cat_wise_ssize]
        self.sample_weight = 1.0/(np.concatenate(sample_weight,axis=0)+self.data_args["epsilon"])

        print("sample weights for each category:\n",
                [
                    [1.0/((cat_ssize/total_examples)+(self.data_args["epsilon"]))] 
                        for cat_ssize in cat_wise_ssize
                ]
        )


if __name__=="__main__":
    #Creating the data handler
    data_args={}
    data_args["max_len"]=100        
    data_args["emb_path"]="random"
    data_args["train_split"]=0.8
    data_args["epsilon"] = 1e-3         #for numerical stability in sample weights
    data_handle = DataHandler(data_args)

    #Now creating our dataset from domain1 (original sentiment)
    # domain1_path = "counterfactually-augmented-data-master/sentiment/orig/"
    # data_handle.data_handler_ltdiff_paper_sentiment(domain1_path)

    #Getting the dataset from amaon reviews 
    cat_list = ["beauty","software","appliance","faishon","giftcard","magazine"]
    path = "dataset/amazon/"
    data_handle.data_handler_amazon_reviews(path,cat_list,1000)
    # pdb.set_trace()






    
