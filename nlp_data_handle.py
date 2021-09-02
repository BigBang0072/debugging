import numpy as np
import pandas as pd 
import tensorflow as tf
import gensim.downloader as gensim_api

import string
import spacy
nlp = spacy.load("en_core_web_lg")
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

import os
import gzip
import json
import re 
import pprint
import pdb
pp=pprint.PrettyPrinter(indent=4)
import random

from transformers import AutoTokenizer

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


class DataHandleTransformer():
    '''
    This class will serve the experiment ground for data handling for
    all transformer related experiments.
    '''
    def __init__(self,data_args):
        self.data_args = data_args
        self.tokenizer = AutoTokenizer.from_pretrained(data_args["transformer_name"])

        self.delimiter=",|\?|\!|-|\*| |  |;|\.|\(|\)|\n|\"|:|'|/|&|`|[|]|\{|\}|\>|\<"
    
    def _clean_the_document(self,document):
        #Splitting the document by the delimiters
        tokens = re.split(self.delimiter,document)
        
        #Updating the vocablury dictionary
        doc2idx = []
        for token in tokens:
            if(len(token)==0):
                continue
            token = token.lower()

            # #Adding the token to vocab
            # if token not in self.dict_w2i:
            #     self._add_word_to_vocab(token)
            
            #Adding the token idx if not approached max len
            if self.data_args["max_len"]==len(doc2idx):
                break
            doc2idx.append(token)
        

        #TODO: Later we could remove some of the words --> unk based on frequency
        return doc2idx,len(tokens)
    
    def _spacy_cleaner(self,doc):
        '''
        This function will clean the document using the spacy's functions
        and return the processed doc
        '''
        punctuations = string.punctuation
        #Lets not remove stopwords for now, because it could also be a possible bug
        # stopwords = list(STOP_WORDS)
        parser = English()

        #Creating the tokens
        tokens = nlp(doc)
        tokens = [word.lemma_.lower().strip() 
                    if word.lemma_ !="-PRON-" 
                    else word.lower_ 
                        for word in tokens
        ]
        tokens = [word for word in tokens if word not in punctuations]

        processed_doc = " ".join([i for i in tokens])

        return processed_doc
    
    def _get_topic_labels(self,all_cat_df,pdoc_name,num_topics,topic_col_name):
        '''
        This will convert all the documents into a topic model
        and label the documents accordinly.
        '''
        #Getting the list of documents
        pdoc_list =[]
        for cat_df in all_cat_df.values():

            cat_pdocs = cat_df[pdoc_name].tolist()
            pdoc_list+=cat_pdocs
        
        #Vectorizing the data
        vectorizer = CountVectorizer(min_df=0.01, 
                                        max_df=0.99, 
                                        stop_words=None, 
                                        lowercase=True,
        )
        data_vectorized = vectorizer.fit_transform(pdoc_list)

        #Performing the LDA
        lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, 
                                        learning_method='online',verbose=True)
        data_lda = lda.fit_transform(data_vectorized)

        #Saving the topic distribution
        self._save_topic_distiribution(lda,vectorizer,"lda_topics.txt",top_n=30)

        #Now we have to label the documents with topic
        for cat_df in all_cat_df.values():
            topic_label_list = []
            for ctidx in range(cat_df.shape[0]):
                doc = cat_df.iloc[ctidx][pdoc_name]
                #Getting the topic distriubiton
                doc_vector = vectorizer.transform([doc])
                doc_topic = np.array(lda.transform(doc_vector)[0])

                #Creating the doc_topic_label
                doc_topic_label = np.stack([doc_topic,1-doc_topic],axis=-1).tolist()
                topic_label_list.append(doc_topic_label)

            #Assigning the topic label in the new column
            cat_df[topic_col_name]=topic_label_list
        
        return all_cat_df

    def _save_topic_distiribution(self,model, vectorizer,fname, top_n=100):
        #Saving the topic distribution to a file
        file_path = "nlp_logs/{}/".format(self.data_args["expt_name"])
        os.makedirs(file_path,exist_ok = True)

        #Writing to the file
        with open(file_path+fname,"w") as whandle:
            for idx, topic in enumerate(model.components_):
                feature_vector = [
                                    (vectorizer.get_feature_names()[i], "{:.5f}".format(topic[i]))
                                        for i in topic.argsort()[:-top_n - 1:-1]
                                ]
                write_string = "Topic:{}\t\tComponent:{}".format(idx,feature_vector)
                print("Topic:{} \t\t Component:{}".format(idx,feature_vector))

    def _convert_df_to_dataset(self,df,doc_col_name,label_col_name,topic_col_name):
        '''
        This function will conver the dataframe to a tensorflow dataset
        to be used as input for Transformer.
        '''
        #First of all parsing the review documents into tensors
        doc_list = []
        label_list = []
        topic_list = []
        for ridx in range(df.shape[0]):
            doc_list.append(df.iloc[ridx][doc_col_name])
            label_list.append(df.iloc[ridx][label_col_name])
            topic_list.append(df.iloc[ridx][topic_col_name])
        
        #Now we will parse the documents
        encoded_doc = self.tokenizer(
                                    doc_list,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
        input_idx = encoded_doc["input_ids"]
        attn_mask = encoded_doc["attention_mask"]

        #Creating the dataset for this category
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=np.array(label_list),
                                    topic=np.array(topic_list),
                                    input_idx = input_idx,
                                    attn_mask = attn_mask
                                )
        )

        return cat_dataset

    def amazon_reviews_handler(self,):
        '''
        This function will handle all the data processing needed for
        handling the amazon reviews.
        '''
        #Parser for the zipped file
        def parse(path):
            g = gzip.open(path,"rb")
            for l in g:
                yield json.loads(l)

        def get_category_df(path,cat,num_sample,cidx):
            pos_list = []
            neg_list = []
            pos_doclen = []
            neg_doclen = []
            skip_count = 0
            topic_count=0
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
                sentiment = 1 if d["overall"]>3 else 0

                #Also we would like to get the topic for this document
                # doc_tokens,doclen = self._clean_the_document(d["reviewText"])

                # genre_words  = set(["science","tragedy","drama","comedy","fiction","fantasy","horror","cartoon"])
                # topic=0
                # if(genre_words.intersection(set(doc_tokens))!=0):
                #     topic=1
                #     topic_count+=1

                
                if(len(neg_list)<num_sample and sentiment==0):
                    processed_doc = self._spacy_cleaner(d["reviewText"])
                    neg_list.append((sentiment,d["reviewText"],processed_doc))
                    neg_doclen.append(len(d["reviewText"]))
                elif (len(pos_list)<num_sample and sentiment==1):
                    processed_doc = self._spacy_cleaner(d["reviewText"])
                    pos_list.append((sentiment,d["reviewText"],processed_doc))
                    pos_doclen.append(len(d["reviewText"]))
                
                
            
            print("===========================================")
            #Getting the stats from this category
            print("cat:{}\tpos:{}\tneg:{}\tpos_avlen:{}\tneg_avlen:{}\tskips:{}".format(
                                        cat,
                                        len(pos_list),
                                        len(neg_list),
                                        np.mean(pos_doclen),
                                        np.mean(neg_doclen),
                                        skip_count,
            ))

            #Now we will create the dataframe for this category
            all_data_list = pos_list+neg_list
            random.shuffle(all_data_list)
            label, doc, pdoc = zip(*all_data_list)
            topic = [cidx]*len(label)

            #Creating the dataframe
            cat_df = pd.DataFrame(
                            dict(
                                label=label,
                                topic=topic,
                                doc=doc,
                                pdoc=pdoc,
                            )
            )
            print("Created category df for: ",cat)
            print(cat_df.head())
            print("===========================================")
            return cat_df
        
        #Getting the dataframe for each categories
        all_cat_df = {}
        for cidx,cat in enumerate(self.data_args["cat_list"]):
            cat_df =  get_category_df(
                                self.data_args["path"],
                                cat,
                                self.data_args["num_sample"], 
                                cidx,                        
            )
            all_cat_df[cat]=cat_df


        #Getting the topic for each of dataframe
        all_cat_df = self._get_topic_labels(all_cat_df=all_cat_df,
                                            pdoc_name="pdoc",
                                            num_topics=self.data_args["num_topics"],
                                            topic_col_name="topic")


        #Creating the zipped dataset
        all_cat_ds = {}
        for cat in self.data_args["cat_list"]:
            cat_df = all_cat_df[cat]
            #Getting the dataset object
            cat_ds = self._convert_df_to_dataset(
                                        df=cat_df,
                                        doc_col_name="doc",
                                        label_col_name="label",
                                        topic_col_name="topic"
            )
            all_cat_ds[cat]=cat_ds
        #Merging into one dictionary
        dataset = tf.data.Dataset.zip(all_cat_ds)
        
        #Now its time to build a data loader for the transformer
        # def df_iterator(df,name):
        #     num_example = df.shape[0]
        #     idx = 0 
        #     while True:
        #         label = df.iloc[idx]["label"]
        #         doc = df.iloc[idx]["doc"]

        #         #Now parsing the document from the tokenizer
        #         encoded_doc = self.tokenizer(
        #                                 doc,
        #                                 padding='max_length',
        #                                 truncation=True,
        #                                 max_length=self.data_args["max_len"],
        #                                 return_tensors="tf"
        #         )
        #         input_idx = encoded_doc["input_ids"]
        #         attn_mask = encoded_doc["attention_mask"]


        #         idx = (idx+1)%num_example
        #         if(idx==0):
        #             print("New Start df: ",name)
                
        #         yield (label,input_idx,attn_mask)
        

        # def get_next_item(all_cat_dfs,all_cat_names):
        #     #Creating the dataset from iteratable object
        #     all_cat_iter = {
        #         cat: df_iterator(all_cat_dfs[cat],cat)
        #             for cat in all_cat_names
        #     }

        #     #Now one by one we will yield one example from each
        #     idx = 0 
        #     while True:
        #         all_cat_samples = [
        #             next(all_cat_iter[cat])
        #                 for cat in all_cat_names
        #         ]

        #         yield all_cat_samples

        #         idx+=1
        #         #Getting out of this loop once we iterated over all the examples
        #         '''
        #         The side effect is if some of the ds has lesser num of example
        #         then they will be cycled.
        #         '''
        #         if(idx==(self.data_args["num_sample"]*self.data_args["num_class"])):
        #             break

        # #Now creating the dataset from this iterator
        # dataset = tf.data.Dataset.from_generator(
        #     get_next_item,
        #     output_signature=(
        #         (
        #             tf.TensorSpec(shape=(),dtype=tf.int32),
        #             tf.TensorSpec(shape=(self.data_args["max_len"]),dtype=tf.int32),
        #             tf.TensorSpec(shape=(self.data_args["max_len"]),dtype=tf.int32),
        #         )
        #     )
        # )


        dataset = dataset.shuffle(self.data_args["shuffle_size"])
        dataset = dataset.batch(self.data_args["batch_size"])
        return dataset


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






    
