from base64 import encode
from cProfile import label
from collections import defaultdict
import numpy as np
import pandas as pd 
import tensorflow as tf
import gensim.downloader as gensim_api

import string
# import spacy
# nlp = spacy.load("en_core_web_lg",disable=["tagger", "parser", "lemmatizer", "ner", "textcat"])
# from spacy.lang.en.stop_words import STOP_WORDS
# from spacy.lang.en import English
# from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
# from sklearn.feature_extraction.text import CountVectorizer

# from sklearn.neighbors import NearestNeighbors

import os
import gzip
import json
import re 
import pprint
import pdb
pp=pprint.PrettyPrinter(indent=4)
import random
import jsonlines
from tqdm import tqdm

#Setting the random seed
# random.seed(22)
# np.random.seed(22)

from transformers import AutoTokenizer, RobertaTokenizer

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
        #Setting the random seed
        random.seed(22)
        np.random.seed(22)

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
        #Setting the seed for the furthur operation
        tf.random.set_seed(data_args["run_num"])
        random.seed(data_args["run_num"])
        np.random.seed(data_args["run_num"])

        if "roberta" in data_args["transformer_name"]:
            self.tokenizer = RobertaTokenizer.from_pretrained(data_args["transformer_name"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(data_args["transformer_name"])

        self.delimiter=",|\?|\!|-|\*| |  |;|\.|\(|\)|\n|\"|:|'|/|&|`|[|]|\{|\}|\>|\<"
        self.word_count=defaultdict(int)
        self.filter_dict=None

        self.emb_model=None
        self.vocab_w2i=None
    
    def _clean_the_document(self,document):
        #Splitting the document by the delimiters
        tokens = re.split(self.delimiter,document)
        
        #Updating the vocablury dictionary
        doc2idx = []
        token_list = []
        for token in tokens:
            if(len(token)==0):
                continue
            token = token.lower()

            # #Adding the token to vocab
            self.word_count[token]+=1
            
            #Adding the token idx if not approached max len
            if self.data_args["max_len"]==len(doc2idx):
                break
            doc2idx.append(token)
            token_list.append(token)
        

        #TODO: Later we could remove some of the words --> unk based on frequency
        return doc2idx,len(tokens),token_list
    
    def _spacy_cleaner(self,doc,tfreq_ulim):
        '''
        This function will clean the document using the spacy's functions
        and return the processed doc
        '''
        # punctuations = string.punctuation
        #Lets not remove stopwords for now, because it could also be a possible bug
        # stopwords = list(STOP_WORDS)
        filter_dict = self._get_filter_word_dict(tfreq_ulim)

        #Creating the tokens
        tokens = nlp(doc)
        tokens = [word.lemma_.lower().strip() 
                    if word.lemma_ !="-PRON-" 
                    else word.lower_ 
                        for word in tokens
        ]
        tokens = [word for word in tokens if word not in filter_dict]

        processed_doc = " ".join([i for i in tokens])

        return processed_doc
    
    def _get_filter_word_dict(self,tfreq_ulim):
        '''
        This function will get the list of words whcih we want to filter
        ie.  ones >= bottom total * tf_ulim words
        '''
        if self.filter_dict !=None:
            return self.filter_dict
        
        word_tf=None
        if(type(tfreq_ulim)==type(1)):
            word_tf = [(word,freq) for word,freq in self.word_count.items() if freq>=tfreq_ulim]
            word_tf_keep = [(word,freq) for word,freq in self.word_count.items() if freq<tfreq_ulim]
            self.vocab_dict={word:idx for idx,(word,_) in enumerate(word_tf_keep)}
        else:
            word_tf = [(word,freq) for word,freq in self.word_count.items()]
            word_tf.sort(key=lambda x:x[-1])
            slice_idx = int(tfreq_ulim*len(word_tf))
            word_tf=word_tf[slice_idx:]
            # pdb.set_trace()
            self.vocab_dict={word:idx for idx,(word,_) in enumerate(word_tf[0:slice_idx])}

        #Hasing the words to remove
        filter_dict = {}
        for word,_ in word_tf:
            filter_dict[word]=True

        #Adding the stop words too in this list
        stopwords=list(STOP_WORDS)
        for word in stopwords:
            filter_dict[word]=True
        
        self.filter_dict=filter_dict

        return filter_dict,self.vocab_dict
    
    def _get_topic_labels_lda(self,all_cat_df,pdoc_name,num_topics,topic_col_name):
        '''
        This will convert all the documents into a topic model
        and label the documents accordinly.
        '''
        #Getting the list of documents
        pdoc_list =[]
        for cat_df in all_cat_df.values():
            for idx in range(cat_df.shape[0]):
                doc = cat_df.iloc[idx][pdoc_name]
                # cat_pdoc = self._spacy_cleaner(doc,tfreq_ulim=0.7)
                pdoc_list.append(doc)
        
        #Generating the vocabulary
        _,vocab_dict=self._get_filter_word_dict(tfreq_ulim=self.data_args["tfreq_ulim"])

        #Vectorizing the data
        vectorizer = CountVectorizer(min_df=self.data_args["min_df"], 
                                        max_df=self.data_args["max_df"], 
                                        stop_words=None, 
                                        lowercase=True,
                                        vocabulary=vocab_dict
        )
        data_vectorized = vectorizer.fit_transform(pdoc_list)


        #Performing the LDA
        lda = LatentDirichletAllocation(n_components=num_topics, 
                                        max_iter=self.data_args["lda_epochs"], 
                                        learning_method='online',verbose=True)
        data_lda = lda.fit_transform(data_vectorized)

        #Saving the topic distribution
        self._save_topic_distiribution(lda,vectorizer,"lda_topics.txt",top_n=50)

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
    
    def _get_topic_labels_manual(self,all_cat_df,doc_col_name,pdoc_col_name,label_col_name):
        '''
        Here we will try to manually annotate certain concepts using bag of words
        approach which we think are clean and have clear distinction ob being 
        spurious and causal.

        new_cat_df = 
        {
                    doc             : the actual document of this category/domain
                    pdoc            : the procssed/tokenized doc for all the corresponding doc
                    topic_feature   : the topic annotation to be used directly as a feature
                                        contains the tf of word which overlap with topic BOW.
                    label           : this is the main task label eg. sentiment classification etc
        }
        '''
        #First of all we get the topic list
        if(self.data_args["extend_topic_set"]==True):
            self._load_word_embedding_for_nbr_extn()
            self._create_topic_list(extend=True)
        else:
            self._create_topic_list(extend=False)


        complete_topic_label_list = []
        complete_topic_list = []
        complete_doc_list = []
        new_all_cat_df = {}
        for cat_name,cat_df in all_cat_df.items():
            topic_label_list = []
            cat_doc_list=[]
            cat_pdoc_list=[]
            cat_topic_feature_list = [] #using topic as a feature
            cat_label_list = []
            for ctidx in range(cat_df.shape[0]):
                pdoc = cat_df.iloc[ctidx][pdoc_col_name]
                cat_pdoc_list.append(pdoc)
                cat_doc_list.append(cat_df.iloc[ctidx][doc_col_name])
                
                #Getting the topic distriubiton
                doc_topic = self._get_topic_annotation_manual(pdoc)

                #Saving the topic labels as a feature
                cat_topic_feature_list.append(doc_topic)
                cat_label_list.append(cat_df.iloc[ctidx][label_col_name])

                #Adding each of the topic and label separately
                for topic_idx in range(doc_topic.shape[0]):
                    #Adding the topic label
                    if(doc_topic[topic_idx]==1.0):
                        complete_topic_label_list.append(0)
                    else:
                        complete_topic_label_list.append(1)
                    
                    #Now adding the topic number and doc
                    complete_topic_list.append(topic_idx)
                    complete_doc_list.append(cat_df.iloc[ctidx][doc_col_name])

                #Creating the doc_topic_label
                # doc_topic_label = np.stack([doc_topic,1-doc_topic],axis=-1).tolist()
                # topic_label_list.append(doc_topic_label)
                # complete_topic_label_list.append(doc_topic_label)

            #Assigning the topic label in the new column
            # cat_df[topic_col_name]=topic_label_list

            # #Getting the topic distirbution within each of the category
            # topic_label_arr = np.sum(np.array(topic_label_list),axis=0).tolist()
            # print("Topic Distribution for cat: ",cat_name)
            # pp.pprint(topic_label_arr)

            #Creating the new dataframe for the cat
            new_cat_df = pd.DataFrame(
                            {
                                doc_col_name:cat_doc_list,
                                pdoc_col_name : cat_pdoc_list,
                                "topic_feature":cat_topic_feature_list,
                                "label":cat_label_list
                            }
            )
            new_all_cat_df[cat_name]=new_cat_df
        
        #Now we have all the examples labelled with all the topic
        merged_topic_df = pd.DataFrame(
                                {
                                    "topic":complete_topic_list,
                                    "label":complete_topic_label_list,
                                    doc_col_name:complete_doc_list
                                }
        )

        all_topic_df = {}
        #Getting the topic dataframe
        for tidx,topic_df in merged_topic_df.groupby("topic"):
            all_topic_df[tidx]= self._get_topic_df(tidx,topic_df)
            
        
        return all_topic_df,new_all_cat_df

        #Getting the complete topic label list
        # complete_topic_label_arr = np.sum(
        #                                 np.array(complete_topic_label_list),
        #                                 axis=0
        #                         )
        # complete_topic_label_total = np.sum(complete_topic_label_arr,axis=-1,keepdims=True)
        # topic_label_weights = complete_topic_label_arr/complete_topic_label_total
        # pp.pprint("Topic Weights:")
        # pp.pprint(topic_label_weights)

        # #Assigning this weight list to all the dataframes
        # for cat_name,cat_df in all_cat_df.items():
        #     print("Assignign the sample weight in cat_df: ",cat_name)
        #     topic_weight_list= []
        #     for ctidx in range(cat_df.shape[0]):
        #         #Getting the label assigned to this example
        #         topic_label = np.argmax(cat_df.iloc[ctidx][topic_col_name],axis=-1).tolist()
        #         topic_idx = range(len(topic_label))

        #         #Now we will give the sample weight based on the label for each topic
        #         topic_weight = topic_label_weights[topic_idx,topic_label]
        #         #Reverting the weights to be opposite of class raito
        #         topic_weight = 1 - topic_weight
        #         # topic_weight = np.expand_dims(topic_weight,axis=-1)

        #         topic_weight_list.append(topic_weight)  #dim = (num_topic)
            
        #     #Now we will assign the label weights to this df
        #     cat_df[topic_weight_col_name]=topic_weight_list


        
        # return all_cat_df
        return
    
    def _get_topic_df(self,tidx,topic_df):
        '''
        This function will develop topic dataset for this topic
        '''
        #Shuffling the dataframe so that we have even dist from diff category of data
        #But this should't matter much
        # topic_df = topic_df.sample(frac=1).reset_index(drop=True)

        class0_df = topic_df[topic_df["label"]==0]
        class1_df = topic_df[topic_df["label"]==1]

        num_class0 = class0_df.shape[0]
        num_class1 = class1_df.shape[0]
        min_num = min(num_class0,num_class1)
        if(self.data_args["num_topic_samples"]!=None):
            min_num = min(self.data_args["num_topic_samples"],min_num)
        print("##############################")
        print("class0:{}\tclass1:{}\tmin_num:{}".format(num_class0,num_class1,min_num))

        class0_df = class0_df.iloc[0:min_num]
        class1_df = class1_df.iloc[0:min_num]

        topic_df = pd.concat([class0_df,class1_df],axis=0,ignore_index=True)
        topic_df = topic_df.sample(frac=1,random_state=22).reset_index(drop=True)

        print("New topic df created: {}\tshape:{} ".format(tidx,topic_df.shape[0]))
        print(topic_df.head())

        #Now creating the dataset object from here
        return topic_df
    
    def _create_topic_list(self,extend):
        '''
        '''
        #Should have high importance
        pos_adjective = set([
            "good","great","awesome","wonderful","terrific","graceful",
            "ecstatic", "lucid", "extraordinary", "impressive", "spectacular",
            "happy", "delicious", "promoted", "powerful", "attractive",
            "love", "outstanding", "intelligent", "tough", "brave", "gentleman",
            "strong", "loving", "caring", "gentle", "well", "valuable", "pretty",
            "trusted", "popular", "loyal", "interesting", "important", "nice",
            "smart", "finest", "talented", "charismatic", "respectful"
        ])
        neg_adjective  = set([
            "bad", "dreadful", "awful", "horrible", "horrific", "terrible",
            "grim", "boring", "ghastly", "foul", "icky", "lousy", "unhappy",
            "unworthy"
        ])
        negations = set([
            "not", "didnt", "did", "was", "wasnt", "is", "isnt"
        ])
        adverbs = set([
            "regularly", "mostly", "terribly", "happily", "briefly", "boldly", 
            "solidly", "heavily", "interestingly", "unfortunately", "fortunately", 
            "badly", "quickly", "incredibly"
        ])

        #Should have low importance
        religion = set([ 
            "islam", "jew", "jews", "fundamantalist", "hindu", "muslim", "christian", 
            "atheist", "buddha", "jain", "sikh", "religion", "faith", "worship", 
            "judaism", "hinduism", "christianity", "buddhism", "monk" 
        ])

        pos_gender= set([ 
            "male","man","john","he","him","his","guy","boy","harry",
            "adam","steve","actor","husband","father","uncle",
            "gentleman","masculine","chap","bloke","lad","dude",
            "bro","brother","son"
        ])

        electronics = set([ 
            "phone", "computer", "circuit", "battery", "television", "remote",
            "headset", "charger", "telephone", "radio", "antenna", "tower", "signal",
            "screen", "keboard", "mouse", "keypad", "desktop", "fan", "ac", "cooler",
            "wire", "solder"
        ])

        pronoun = set([ 
            "he", "she", "it", "they", "them","i", "we", "themselves", "thy","thou",
            "her","him",
        ])

        kitchen = set([ 
            "lipton", "tea", "taste", "food", "jam", "kitchen", "rice", "meal", "cook",
            "cooker", "bowl", "herbs", "herb", "fruits", "vegetable", "apple", "orange",
            "grapes", "guava"
        ])

        genre = set([ 
            "horror", "comedy", "fantasy", "thriller", "rock", "pop", "romantic", "romcom",
            "suspense", "war", "worldwar", "documentry", "fiction", 
            "poetry", "novel", "mystery", "detective", "crime", "dystopian", "western", 
            "history"
        ])

        #Domain specific topics itself
        arts = set([
            "painting","picture","sculpture","artistic","arts",
            "painter", "color"
        ])

        books = set([
            "book", "cover", "binding", "author", "novel", "comic", 
            "stationary", "poet"
        ])

        clothes = set([ 
            "clothes", "faishon", "jeans", "shirt", "frock", "pant", 
            "costume", "tie", "belt"
        ])

        groceries = set([ 
            "groceries", "rice", "vegetable", "oil", "beef", "milk", "bread",
            "pasta", "corn", "dairy"
        ])

        pos_movies = set([ 
            "romantic","comedy","romcom","action","matrix","inception",
            "titanic","kung","fu","genius","harry","potter","shawshank",
            "redemption","spiderman","hulk","knight","gump","ryan",
            "interstellar","joker","batman","superman","avengers",
            "300","prestige"
        ])

        pets = set([ 
            "pet", "dog", "cat", "snake", "rabbit", "parrot", "hamster",
            "owl", "lizard", "store"
        ])

        phone = set([ 
            "phone","mobile", "samsung", "nokia", "blackberry", "telephone",
            "sim", "battery","charger", "cable"
        ])

        tools = set([ 
            "tool","hacksaw","knife", "saw", "screw", "blower", "hammer",
            "blade", "bolt", "keys" ,"key"
        ])


        #Toy Dataset specific topics
        male = set([ 
            "man","adam"
        ])
        female = set([ 
            "women","eve"
        ])
        white = set([ 
            "white","john"
        ])
        black = set([ 
            "black","smith",
        ])
        straight = set([ 
            "straight"
        ])
        gay = set([ 
            "gay"
        ])


        #Creating the topic list
        old_topic_list = [
            pos_adjective,
            #neg_adjective,#negations,adverbs,
            #religion,
            pos_gender,#,electronics,
            #pronoun,#,kitchen,genre,
            #arts,books,
            #clothes#,groceries,
            pos_movies#,pets,phone,tools,
            #male,female,white,black,straight,gay,
        ]

        if(extend==True):
            new_topic_list = []
            for topic_set in old_topic_list:
                print("\n\nGetting the newly extended topic set")
                print("InitialSet:\n ",topic_set)
                #First of all getting the new topic set 
                topic_set = self._extend_topic_set_wordembedding(topic_set)
                print("ExtendedSet:\n",topic_set)
                new_topic_list.append(topic_set)
            
            self.topic_list = new_topic_list
        else:
            self.topic_list = old_topic_list
        
        #Updating the number of topics present
        self.data_args["num_topics"]=len(self.topic_list)

    def _get_topic_annotation_manual(self,pdoc):
        '''
        '''
        topic_list = self.topic_list
        self.data_args["num_topics"]=len(topic_list)


        #Now we will label the document
        assert type(pdoc) == type([1,2]), "pdoc is not list of words"

        #TODO: counting the word occurance for better
        pdoc_dict = defaultdict(int)
        for wrd in pdoc:
            pdoc_dict[wrd]+=1

        #Now getting the topic distribution for this document
        pdoc_set = set(pdoc)
        topic_label = []
        for topic_set in topic_list:
            
            #Binary Feature Here we can use them as topic label then otherwise we need to quantize
            # if len(pdoc_set.intersection(topic_set))!=0:
            #     topic_label.append(1.0)
            # else:
            #     topic_label.append(0.0)
            
            #Counting the number of intersection
            # topic_label.append(len(pdoc_set.intersection(topic_set)))

            #Counting the frequency of the number of intersection too
            topic_freq = sum([pdoc_dict[tword] for tword in topic_set])
            topic_label.append(topic_freq)

        return np.array(topic_label)
    
    def _extend_topic_set_wordembedding(self,topic_set):
        '''
        Given the seed words for the topic we will extend this set to
        using the neighborhood of the word2vec embedding hoping that
        they will be useful.
        '''
        #Get the embedding of words in the set
        seed_words = [word for word in topic_set if word in self.vocab_w2i]
        seed_embs = np.stack([self.emb_matrix[self.vocab_w2i[word]] 
                        for word in seed_words
                    ],axis=0)
        
        #Now we will find the nearest neighbors
        print("Finding the neighbors! Sit Tight!")
        n_indices = self.neigh_tree.kneighbors(X=seed_embs,
                                    n_neighbors=self.data_args["num_neigh"],
                                    return_distance=False)
        #Getting the neighbor words for this topic set
        new_topic_set = []
        for zidx in range(n_indices.shape[0]):
            for yidx in range(n_indices.shape[1]):
                n_idx = n_indices[zidx,yidx]
                new_topic_set.append(self.vocab_i2w[n_idx])
        new_topic_set=set(new_topic_set+seed_words)

        return new_topic_set
    
    def _load_word_embedding_for_nbr_extn(self,):
        if self.vocab_w2i!=None:
            return 
        
        #Loading the embedding from the gensim repo
        print("Loading the WordVectors via Gensim! Hold Tight!")
        emb_model = gensim_api.load(self.data_args["emb_path"])

        #Next we will load the smallar subsampled vocablury
        vocab_df = pd.read_csv(self.data_args["vocab_path"],sep="\t",header=0)
        vocab_list = vocab_df["word"].tolist()

        #Now filtering out the required embeddings
        emb_matrix = []
        self.vocab_w2i = {}
        for widx,word in enumerate(vocab_list):
            try:
                word_vec = emb_model.get_vector(word)
                emb_matrix.append(word_vec)
                #Adding the word to vocablury
                self.vocab_w2i[word]=len(self.vocab_w2i)
            except:
                print("Missed Word: ",word)
        #Now we will assign this matrix to the class
        self.emb_matrix = np.stack(emb_matrix,axis=0)
        self.vocab_i2w = {idx:word for word,idx in self.vocab_w2i.items()}

        #Lets create the neighborhood object too here
        self.neigh_tree = NearestNeighbors(algorithm="auto",
                                        metric="minkowski",#DEBUG
                                        p=2,leaf_size=30,
                                        n_jobs=-1).fit(self.emb_matrix)
    
    def _load_full_gensim_word_embedding(self,):
        '''
        We will load the full given word embedding matrix in the memory.
        This shouldn't be very expensive on memory as the maximum embedding
        matrix size is in few GBs.
        '''
        #Loading the embedding from the gensim repo
        if self.emb_model==None:
            print("Loading the WordVectors via Gensim! Hold Tight!")
            emb_model = gensim_api.load(self.data_args["emb_path"])
            self.emb_model = emb_model

        '''
        This has everything we need already:
        1. index_to_key         : list of words sorted by the index
        2. key_to_index         : word2index dict
        4. get_vector           : one vector for the given word
        5. vectors              : all the embedding matrix
        '''
        return 
        
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
                write_string = "Topic:{}\t\tComponent:{}\n".format(idx,feature_vector)
                whandle.write(write_string)
                print("Topic:{} \t\t Component:{}".format(idx,feature_vector))

    def _convert_df_to_dataset_stage2_NBOW(self,df,pdoc_col_name,label_col_name,
                                        topic_feature_col_name,debug_topic_idx):
        '''
        This function will generate the tf dataset for the stage 2 when we are using 
        the "neural" BOW model.

        So here from the pdoc i.e the tokenized word list we will convert them to the 
        index wrt to the word emebdding vectors which will be decoded in the training
        step directly from the embedding layer.
        '''
        #Loading the gensim embedidng model
        self._load_full_gensim_word_embedding()

        #First of all parsing the review documents into tensors
        doc_list = []
        all_index_list = []
        label_list = []
        topic_label_list = []
        # topic_weight_list = []
        for ridx in range(df.shape[0]):
            #Getting the label
            label_list.append(df.iloc[ridx][label_col_name])

            #Getting the topic labels
            topic_feature = df.iloc[ridx][topic_feature_col_name]
            # topic_label=0
            # if topic_feature[debug_topic_idx]>0:
            #     topic_label=1

            #Keeping the label for all the topics
            topic_label=[]
            for fval in topic_feature:
                if fval>0:
                    topic_label.append(1)
                else:
                    topic_label.append(0)
            topic_label_list.append(topic_label)
            
            # topic_weight_list.append(df.iloc[ridx][topic_weight_col_name])

            #Now processing the document to token idx
            token_list = df.iloc[ridx][pdoc_col_name]
            #Unknown words are being left out all together
            index_list = [
                self.emb_model.key_to_index[token] 
                    for token in token_list
                        if token in self.emb_model.key_to_index
            ]
            #Making all the vector one-sized
            padding = [self.emb_model.key_to_index["unk"]]*(self.data_args["max_len"]-len(index_list))
            index_list = index_list + padding
            all_index_list.append(index_list)

        #Printing the topic label distribution for both main task class
        label_list = np.array(label_list,np.int32)
        topic_label_list=np.array(topic_label_list,np.int32)
        #Getting the main task class mask
        main_class0_mask = (label_list==0)
        main_class1_mask = (label_list==1)
        def get_topic_segmentation(class_mask,topic_label_arr,name,tidx):
            topic_label_class = topic_label_arr[class_mask]
            print("class:{}\ttidx:{}\tnum_topic_0:{}\tnum_topic_1:{}".format(
                                        name,
                                        tidx,
                                        topic_label_class.shape[0]-np.sum(topic_label_class),
                                        np.sum(topic_label_class)
            ))
        
        print("\n\nGetting the topic segmentation")
        for tidx in range(topic_label_list.shape[-1]):
            print("####################################################")
            get_topic_segmentation(main_class0_mask,topic_label_list[:,tidx],"0",tidx)
            get_topic_segmentation(main_class1_mask,topic_label_list[:,tidx],"1",tidx)
        # pdb.set_trace()

        #TODO: Calculate the topic correlation/predictive correlation with the main topic
        #Should this be symmetric or directional correlation?



        #Creating the dataset for this category
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=label_list,
                                    input_idx = np.array(all_index_list,np.int32),
                                    # topic=np.array(topic_list),
                                    # topic_weight=np.array(topic_weight_list),
                                    # input_idx = input_idx,
                                    # attn_mask = attn_mask,
                                    # topic_feature=topic_feature,
                                    topic_label = topic_label_list
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

        return cat_dataset
    
    def _convert_df_to_dataset_stage2_transformer_syn(self,df,doc_col_name,label_col_name,
                                        topic_feature_col_name,debug_topic_idx):
        '''
        This function will conver the dataframe to a tensorflow dataset
        to be used as input for Transformer.

        This will be specifically be used for stage to when we are training 
        normal model, which will be BERT for our use case.

        Current we will only allow one topic to be debugged at a time for the debugger
        TODO: add multi-topic debugging support.
        '''
        #First of all parsing the review documents into tensors
        doc_list = []
        label_list = []
        topic_label_list = []
        # topic_weight_list = []
        for ridx in range(df.shape[0]):
            #Removing the topic from the input and checking if removal is TP or FP
            doc = df.iloc[ridx][doc_col_name]
            # topic_words = self.topic_list[-1]
            # for word in topic_words:
            #     doc = doc.replace(word,"")


            #Getting the topic labels
            # topic_feature = df.iloc[ridx][topic_feature_col_name]
            # topic_labels=[]
            # for fval in topic_feature:
            #     if fval>0:
            #         topic_labels.append(1)
            #     else:
            #         topic_labels.append(0)
            # topic_label_list.append(topic_labels)

            #Adding the topic manually
            self.data_args["num_topics"]=0
            topic_labels=[] 

            #Adding the first topic [numbered vs no numbered word]
            # self.data_args["num_topics"]+=1
            # topic0_pval = self.data_args["topic_corr_list"][0]
            # topic0_cpd = np.array(
            #     [ 
            #         [topic0_pval,1-topic0_pval],
            #         [1-topic0_pval,topic0_pval],
            #     ]
            # )
            # topic0_cat,doc = self._add_synthetic_topics(
            #                         tidx=0,
            #                         doc=doc,
            #                         label=df.iloc[ridx][label_col_name],
            #                         cpd=topic0_cpd
            # )
            # topic_labels.append(topic0_cat)

            self.data_args["num_topics"]=3
            pval = self.data_args["topic_corr_list"][0]
            cpd = np.array([ 
                [(1-pval)/2,(pval/2),(pval/2),(1-pval)/2],
                [(pval/2),(1-pval)/2,(1-pval)/2,(pval/2)],
            ])
            two_topic_label_list,doc = self._add_synthetic_topics(
                                    tidx="two_topic",
                                    doc=doc,
                                    label=df.iloc[ridx][label_col_name],
                                    cpd=cpd
            )
            #Adding the main topic also as separete topic
            topic_labels=[df.iloc[ridx][label_col_name]]+two_topic_label_list




            #Adding everything to the final list
            doc_list.append(doc)
            label_list.append(df.iloc[ridx][label_col_name])
            topic_label_list.append(topic_labels)
            # topic_weight_list.append(df.iloc[ridx][topic_weight_col_name])
        
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
                                    label=np.array(label_list,dtype=np.int32),
                                    # topic=np.array(topic_list),
                                    # topic_weight=np.array(topic_weight_list),
                                    input_idx = input_idx,
                                    attn_mask = attn_mask,
                                    # topic_feature=topic_feature,
                                    topic_label = np.array(topic_label_list,dtype=np.int32)
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

        #Calculating the predictive performance
        label=np.array(label_list,dtype=np.int32)
        topic_label = np.array(topic_label_list,dtype=np.int32)
        print("\n\nIndividual Distribution:")
        print("main:\tclass0:{}\tclass1:{}".format(
                            np.sum(label==0),
                            np.sum(label==1)
        ))
        for tidx in range(self.data_args["num_topics"]):
            print("topic-{}:\tclass0:{}\tclass1:{}".format(
                                        tidx,
                                        np.sum(topic_label[:,tidx]==0),
                                        np.sum(topic_label[:,tidx]==1)
            ))


        
        #Next we will print the conditional distribution:
        print("\n\nConditional Distribution")
        #Getting the main task class mask
        main_class0_mask = (label==0)
        main_class1_mask = (label==1)
        def get_topic_segmentation(class_mask,topic_label,name,tidx):
            topic_label_class = topic_label[class_mask]
            print("class:{}\ttidx:{}\tnum_topic_0:{}\tnum_topic_1:{}".format(
                                        name,
                                        tidx,
                                        topic_label_class.shape[0]-np.sum(topic_label_class),
                                        np.sum(topic_label_class)
            ))
        
        print("\n\nGetting the topic segmentation")
        for tidx in range(topic_label.shape[-1]):
            print("####################################################")
            get_topic_segmentation(main_class0_mask,topic_label[:,tidx],"0",tidx)
            get_topic_segmentation(main_class1_mask,topic_label[:,tidx],"1",tidx)


        return cat_dataset
    
    def _add_synthetic_topics(self,tidx,doc,label,cpd):
        '''
        This function will take a document and a label and based 
        on the conditional distribution arr, it will assign the 
        given topic (tidx)'s category to the current doc.

        input:
        tidx:   the topic we want to insert
        doc:    input doc where we want to insert the topic
        label:  current label of the doc
        cpd:    the conditional distribution of the label with topic
                    this is full conditional prob matrix
                    row is for the given label the dist of topic category

        returns:
        topic_label : the cateogry of the given topic assigned
        new_doc     : the topic added doc
        '''
        if tidx==0:
            #This is the number word topic
            number_words = [
                "one","two","three","four","five","six","seven","eight","nine","ten",
                "eleven","twelve","thirteen","fourteen","fifteeen","sixteen","seventeen",
                "eighteen","twenty","thirty","fourty","fifty","sixty","seventy","eighty",
                "ninety","hundred","thousand"
            ]

            #Sampling the topic category
            tcat = np.random.choice([0,1],size=1,p=cpd[label,:])[0]
            #Add the topic if we got the signal
            if tcat==1:
                rword = np.random.choice(number_words,size=1)[0]
                new_doc = rword + " " + doc
            else:
                new_doc=doc
            
            return tcat,new_doc
        
        elif tidx=="two_topic":
            #Here we will add two topic at a time. 
            #TODO: Think of a better way
            number_words = [
                "one","two","three","four","five","six","seven","eight","nine","ten",
                "eleven","twelve","thirteen","fourteen","fifteeen","sixteen","seventeen",
                "eighteen","twenty","thirty","fourty","fifty","sixty","seventy","eighty",
                "ninety","hundred","thousand"
            ]
            food_words = [ 
                "ice","bread","icecream","rice","potato","tomato",
                "chocolate","wine","wheat","maize","barley","carrot",
                "palnt","vegan"
            ]

            #Sampling both the topic at once
            tcat = np.random.choice([0,1,2,3],size=1,p=cpd[label,:])[0]
            topic_labels = None
            if tcat==1:
                num_rword = np.random.choice(number_words,size=1)[0]
                new_doc = num_rword + " " + doc
                topic_labels = [1,0]
            elif tcat==2:
                food_rword = np.random.choice(food_words,size=1)[0]
                new_doc = food_rword + " " + doc
                topic_labels = [0,1]
            elif tcat==3:
                num_rword = np.random.choice(number_words,size=1)[0]
                food_rword = np.random.choice(food_words,size=1)[0]
                new_doc = food_rword +" "+ num_rword + " " + doc
                topic_labels = [1,1]
            else:
                topic_labels=[0,0]
                new_doc= doc
            return topic_labels,new_doc

        else:
            raise NotImplementedError()
  
    def _convert_df_to_dataset_stage2_transformer(self,df,doc_col_name,label_col_name,
                                        topic_feature_col_name,debug_topic_idx):
        '''
        This function will conver the dataframe to a tensorflow dataset
        to be used as input for Transformer.

        This will be specifically be used for stage to when we are training 
        normal model, which will be BERT for our use case.

        Current we will only allow one topic to be debugged at a time for the debugger
        TODO: add multi-topic debugging support.
        '''
        #First of all parsing the review documents into tensors
        doc_list = []
        label_list = []
        topic_label_list = []
        # topic_weight_list = []
        for ridx in range(df.shape[0]): 
            doc_list.append(df.iloc[ridx][doc_col_name])
            label_list.append(df.iloc[ridx][label_col_name])

            #Getting the topic labels
            topic_feature = df.iloc[ridx][topic_feature_col_name]
            topic_label=[]
            for fval in topic_feature:
                if fval>0:
                    topic_label.append(1)
                else:
                    topic_label.append(0)
            topic_label_list.append(topic_label)
            
            # topic_weight_list.append(df.iloc[ridx][topic_weight_col_name])
        
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
        label=np.array(label_list,dtype=np.int32)
        topic_label = np.array(topic_label_list,dtype=np.int32)

        #Balancing the dataset based in the last topic now
        total_ltopic_pos = np.sum((topic_label[:,-1]==1))
        total_mneg_take = total_ltopic_pos - np.sum((topic_label[:,-1]==1) * (label==0))
        total_mpos_take = total_ltopic_pos - np.sum((topic_label[:,-1]==1) * (label==1))
        
        mneg_tneg_mask = np.logical_and((label==0),(topic_label[:,-1]==0))
        mpos_tneg_mask = np.logical_and((label==1),(topic_label[:,-1]==0))

        mneg_tpos_mask = np.logical_and((label==0),(topic_label[:,-1]==1))
        mpos_tpos_mask = np.logical_and((label==1),(topic_label[:,-1]==1))

        #Now taking out only relavant portion of data
        input_idx = input_idx[mneg_tpos_mask] + input_idx[mneg_tneg_mask][0:total_mneg_take] \
                    + input_idx[mpos_tpos_mask] + input_idx[mpos_tneg_mask][0:total_mpos_take]
        
        attn_mask = attn_mask[mneg_tpos_mask] + attn_mask[mneg_tneg_mask][0:total_mneg_take] \
                    + attn_mask[mpos_tpos_mask] + attn_mask[mpos_tneg_mask][0:total_mpos_take]
        
        label = label[mneg_tpos_mask] + label[mneg_tneg_mask][0:total_mneg_take] \
                    + label[mpos_tpos_mask] + label[mpos_tneg_mask][0:total_mpos_take]

        topic_label = topic_label[mneg_tpos_mask] + topic_label[mneg_tneg_mask][0:total_mneg_take] \
                    + topic_label[mpos_tpos_mask] + topic_label[mpos_tneg_mask][0:total_mpos_take]


        #Creating the dataset for this category
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=label,
                                    # topic=np.array(topic_list),
                                    # topic_weight=np.array(topic_weight_list),
                                    input_idx = input_idx,
                                    attn_mask = attn_mask,
                                    # topic_feature=topic_feature,
                                    topic_label = topic_label,
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

        #Calculating the predictive performance
        print("\n\nIndividual Distribution:")
        print("main:\tclass0:{}\tclass1:{}".format(
                            np.sum(label==0),
                            np.sum(label==1)
        ))
        for tidx in range(self.data_args["num_topics"]):
            print("topic-{}:\tclass0:{}\tclass1:{}".format(
                                        tidx,
                                        np.sum(topic_label[:,tidx]==0),
                                        np.sum(topic_label[:,tidx]==1)
            ))


        
        #Next we will print the conditional distribution:
        print("\n\nConditional Distribution")
        #Getting the main task class mask
        main_class0_mask = (label==0)
        main_class1_mask = (label==1)
        def get_topic_segmentation(class_mask,topic_label,name,tidx):
            topic_label_class = topic_label[class_mask]
            print("class:{}\ttidx:{}\tnum_topic_0:{}\tnum_topic_1:{}".format(
                                        name,
                                        tidx,
                                        topic_label_class.shape[0]-np.sum(topic_label_class),
                                        np.sum(topic_label_class)
            ))
        
        print("\n\nGetting the topic segmentation")
        for tidx in range(topic_label.shape[-1]):
            print("####################################################")
            get_topic_segmentation(main_class0_mask,topic_label[:,tidx],"0",tidx)
            get_topic_segmentation(main_class1_mask,topic_label[:,tidx],"1",tidx)


        return cat_dataset
    
    def _convert_df_to_dataset_stage1(self,df,doc_col_name,label_col_name,mask_feature_dims):
        '''
        This function will conver the dataframe to a tensorflow dataset
        to be used as input for Transformer.
        '''
        #First of all parsing the review documents into tensors
        doc_list = []
        label_list = []
        # topic_list = []
        # topic_weight_list = []
        for ridx in range(df.shape[0]):
            doc_list.append(df.iloc[ridx][doc_col_name])
            label_list.append(df.iloc[ridx][label_col_name])
            # topic_list.append(df.iloc[ridx][topic_col_name])
            # topic_weight_list.append(df.iloc[ridx][topic_weight_col_name])
        
        #Now we will parse the documents
        # encoded_doc = self.tokenizer(
        #                             doc_list,
        #                             padding='max_length',
        #                             truncation=True,
        #                             max_length=self.data_args["max_len"],
        #                             return_tensors="tf"
        #     )
        # input_idx = encoded_doc["input_ids"]
        # attn_mask = encoded_doc["attention_mask"]

        #Creating the feature array
        topic_feature = np.array(doc_list).astype(np.float32)
        #Makging this feature array to simulate the subset selection
        topic_feature[:,mask_feature_dims] = 0.0


        #Creating the dataset for this category
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=np.array(label_list),
                                    # topic=np.array(topic_list),
                                    # topic_weight=np.array(topic_weight_list),
                                    # input_idx = input_idx,
                                    # attn_mask = attn_mask
                                    topic_feature=topic_feature
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

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
                doc_tokens,doclen,token_list = self._clean_the_document(d["reviewText"])

                # genre_words  = set(["science","tragedy","drama","comedy","fiction","fantasy","horror","cartoon"])
                # topic=0
                # if(genre_words.intersection(set(doc_tokens))!=0):
                #     topic=1
                #     topic_count+=1

                
                if(len(neg_list)<num_sample and sentiment==0):
                    processed_doc = token_list#self._spacy_cleaner(d["reviewText"])
                    neg_list.append((sentiment,d["reviewText"],processed_doc))
                    neg_doclen.append(len(d["reviewText"]))
                elif (len(pos_list)<num_sample and sentiment==1):
                    processed_doc = token_list#self._spacy_cleaner(d["reviewText"])
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
        # all_cat_df = self._get_topic_labels(all_cat_df=all_cat_df,
        #                                     pdoc_name="pdoc",
        #                                     num_topics=self.data_args["num_topics"],
        #                                     topic_col_name="topic")

        #Getting the manually labelled topic
        all_topic_df,new_all_cat_df = self._get_topic_labels_manual(
                                            all_cat_df=all_cat_df,
                                            doc_col_name="doc",
                                            pdoc_col_name="pdoc",
                                            label_col_name="label",
                                            # topic_col_name="topic",
                                            # topic_weight_col_name="topic_weight"
        )


        #Creating the zipped dataset
        all_cat_ds = {}
        # for cat in self.data_args["cat_list"]:
        #     cat_df = new_all_cat_df[cat]
        #     #Getting the dataset object
        #     cat_ds = self._convert_df_to_dataset(
        #                                 df=cat_df,
        #                                 doc_col_name="topic_feature",
        #                                 label_col_name="label",
        #                                 mask_feature_dims=self.data_args["mask_feature_dims"]
        #                                 # topic_col_name="topic",
        #                                 # topic_weight_col_name="topic_weight",
        #     )
        #     all_cat_ds[cat]=cat_ds
        

        #Creating the topic dataset
        all_topic_ds = {}
        # for tidx in self.data_args["topic_list"]:
        #     topic_df = all_topic_df[tidx]
        #     #Getting the dataset object
        #     topic_ds = self._convert_df_to_dataset(
        #                                 df=topic_df,
        #                                 doc_col_name="doc",
        #                                 label_col_name="label",
        #                                 # topic_col_name="topic",
        #                                 # topic_weight_col_name="topic_weight",
        #     )
        #     all_topic_ds[tidx]=topic_ds
        

        return all_cat_ds,all_topic_ds,new_all_cat_df
        
        #Merging into one dictionary
        # dataset = tf.data.Dataset.zip(all_cat_ds)
        
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


        # dataset = dataset.shuffle(self.data_args["shuffle_size"])
        # dataset = dataset.batch(self.data_args["batch_size"])
        # return dataset
        return 

    def toy_nlp_dataset_handler(self,):
        '''
        This function will handle the dataset creation of the toy nlp dataset
        where we will induce the desired bias by ourselves and then test both
        the phases of our algorithm.
        '''
        #First of all we have to load the template dataset
        template_df = self._tnlp_load_template_dataset(self.data_args["task_name"])

        #Now we will create dataset for different domain/category
        all_cat_df = {}
        for cat_name in self.data_args["cat_list"]:
            all_cat_df[cat_name] = self._tnlp_get_category_dataset(
                                        cat_name=cat_name,
                                        causal_ratio=self.data_args["causal_ratio"],
                                        spuriousness_ratio=self.data_args["spurious_ratio"],
                                        template_df=template_df
            )
        
        all_topic_df,new_all_cat_df = self._get_topic_labels_manual(
                                            all_cat_df=all_cat_df,
                                            doc_col_name="doc",
                                            pdoc_col_name="pdoc",
                                            label_col_name="label",
                                            # topic_col_name="topic",
                                            # topic_weight_col_name="topic_weight"
        )
        #Empty for now
        all_cat_ds = {}
        all_topic_ds = {}
        

        return all_cat_ds,all_topic_ds,new_all_cat_df
    
    def toy_nlp_dataset_handler2(self,return_causal=False,return_cf=False,return_fulldict=False):
        '''
        In this handler we will add multiple topics at a time into
        the text and then test the removal and the topic convergence.
        1. Create main topic with p_main (fully predictive)
        2. Create topic_i with p_i correlation with the main topic
        '''
        #Loading the gensim embedidng model
        self._load_full_gensim_word_embedding()

        #Creating the main topic in the dataset
        number_words = [
            "one","two","three","four","five","six","seven","eight","nine","ten",
            "eleven","twelve","thirteen","fourteen","fifteeen","sixteen","seventeen",
            "eighteen","twenty","thirty","fourty","fifty","sixty","seventy","eighty",
            "ninety","hundred","thousand"
        ]
        non_number_words = [
            "nice","device","try","picture","signature","trailer","harry","potter",
            "malfoy","john","switch","taste","glove","baloon", "dog", "horse",
            "switch", "watch", "sun", "cloud", "river", "town", "cow", "shadow",
            "pencil", "eraser"
        ]

        #Setting up for generating the causal data
        init_corr_value=self.data_args["topic_corr_list"][-1]
        if return_causal==True:
            '''
            So, if we try to geenrate the non-causal data, in that case we will 
            have label correlation long with the data will also have that correlation
            with the spurious feature. but again the main classifier is not trained in those
            scenarios so dosent matter

            In other cases even if we generate the non-causal data, we will either
            remove the spurious topic or we will have that single topic-value feature in the
            whole dataset. So, the label should not be predictive
            '''
            #making the data balanced in terms of spurious correlation
            self.data_args["topic_corr_list"][-1]=0.5

        #Creating the examples
        all_example_list = []
        all_example_cfactual_t0 = []    #List of counterfactuals wrt to topic0
        all_example_cfactual_t1 = []    #List of counterfactuals wrt to topic1
        all_example_list_t0_flip = []
        all_example_list_t1_flip = []   #These have the flipped corresponding topic
        all_example_list_only_t0 = []   #These example have spurious feature absent
        all_example_list_t1_is_1 = []   #where the topic feature is set to a value
        all_label_list= []
        for sidx in range(self.data_args["num_sample"]):
            pos_label_list = [1,]
            neg_label_list = [0,]

            #Creating the positive example
            #we wont have any causal fearture now which is fully predictive
            pos_example = ""#"this is a positive template "
            neg_example = ""#"this is a negative example "


            #Creating the topics 1
            tidx0=0
            pos_add_topic0, neg_add_topic0,pos_label_list,neg_label_list =self._generate_topic0_constituents(
                                                number_words=number_words,
                                                non_number_words=non_number_words,
                                                pos_label_list=pos_label_list,
                                                neg_label_list=neg_label_list,
                                                topic0_corr=self.data_args["topic_corr_list"][tidx0],
            )
            
            #Creating the topic 2
            tidx1=1
            pos_add_topic1, neg_add_topic1,pos_label_list,neg_label_list =self._generate_topic1_constituents(
                                                pos_label_list=pos_label_list,
                                                neg_label_list=neg_label_list,
                                                topic1_corr=self.data_args["topic_corr_list"][tidx1]
            )

            #Generating the counterfactuals
            if return_cf==True:
                #Creating the counterfactaul for the positive example
                pos_cfactual_list_topic0,pos_cfactual_list_topic1=self._generate_toy2_counterfactual(
                                                    number_words=number_words,
                                                    non_number_words=non_number_words,
                                                    sentence_prefix=pos_example,
                                                    add_topic0=pos_add_topic0,
                                                    add_topic1=pos_add_topic1,
                                                    label_list=pos_label_list,
                )

                #Creating the counterfactual for the negative example
                neg_cfactual_list_topic0,neg_cfactual_list_topic1=self._generate_toy2_counterfactual(
                                                    number_words=number_words,
                                                    non_number_words=non_number_words,
                                                    sentence_prefix=neg_example,
                                                    add_topic0=neg_add_topic0,
                                                    add_topic1=neg_add_topic1,
                                                    label_list=neg_label_list,
                )

                #Adding the counterfactual
                all_example_cfactual_t0+=[pos_cfactual_list_topic0,neg_cfactual_list_topic0]
                all_example_cfactual_t1+=[pos_cfactual_list_topic1,neg_cfactual_list_topic1]

            #Constructing the examples
            pos_example_main = pos_example + pos_add_topic0 + pos_add_topic1 
            neg_example_main = neg_example + neg_add_topic0 + neg_add_topic1 
            all_example_list+=[pos_example_main,neg_example_main]

            #Constructing the examples by flipping topic0
            pos_example_t0 = pos_example + neg_add_topic0 + pos_add_topic1 
            neg_example_t0 = neg_example + pos_add_topic0 + neg_add_topic1 
            all_example_list_t0_flip+=[pos_example_t0,neg_example_t0]

            #Constructing the examples by flipping topic1
            pos_example_t1 = pos_example + pos_add_topic0 + neg_add_topic1 
            neg_example_t1 = neg_example + neg_add_topic0 + pos_add_topic1 
            all_example_list_t1_flip+=[pos_example_t1,neg_example_t1]

            #Creating the example which just have a topic0
            pos_example_only_t0 = pos_example + pos_add_topic0
            neg_example_only_t0 = neg_example + neg_add_topic0
            all_example_list_only_t0+=[pos_example_only_t0,neg_example_only_t0]

            #Creating the example which have topic1=1 everytime
            pos_example_t1_is_1 = pos_example + pos_add_topic0 + " "+" ".join(["fill"]*10)
            neg_example_t1_is_1 = neg_example + neg_add_topic0 + " "+" ".join(["fill"]*10)
            all_example_list_t1_is_1+=[pos_example_t1_is_1,neg_example_t1_is_1]

            #Creating example with only t1
            # pos_example_only_t1 = pos_example + pos_add_topic1
            # neg_example_only_t1 = neg_example + neg_add_topic1

            #Adding the example and lable
            all_label_list+=[pos_label_list,neg_label_list]


        #Converting the text example to input_idx 
        all_index_arr = self._convert_text_to_widx(all_example_list)
        #Getting the input idx where feature0 is flipped (on-->off or off-->on)
        all_index_arr_t0_flip = self._convert_text_to_widx(all_example_list_t0_flip)
        #Getting the input idx where feature1 is flipped (on-->off or off-->on)
        all_index_arr_t1_flip = self._convert_text_to_widx(all_example_list_t1_flip)
        #Getting the input idx where only feature0 is present
        all_index_arr_only_t0 = self._convert_text_to_widx(all_example_list_only_t0)
        #Getting the input idx where topic1 is set to a fixed label =1
        all_index_arr_t1_is_1 = self._convert_text_to_widx(all_example_list_t1_is_1)

        if return_cf==True:
            #Converting the counterfacutals to the index for topic0
            all_index_arr_cfactual_t0=[]
            for cfactual_list in all_example_cfactual_t0:
                all_index_arr_cfactual_t0.append(self._convert_text_to_widx(cfactual_list))
            all_index_arr_cfactual_t0=np.stack(all_index_arr_cfactual_t0,axis=0)

            #Converting the counterfactuals to index for topic1
            all_index_arr_cfactual_t1=[]
            for cfactual_list in all_example_cfactual_t1:
                all_index_arr_cfactual_t1.append(self._convert_text_to_widx(cfactual_list))
            all_index_arr_cfactual_t1=np.stack(all_index_arr_cfactual_t1,axis=0)

        
        #Creating the dataset object
        all_label_arr = np.array(all_label_list,np.int32)
        self._print_label_distribution(all_label_arr)
        #Shuffling the dataser (no need right now they are balanced)
        #Adding noise to the labels to have non-fully predictive causal features
        all_label_arr_nonoise = all_label_arr.copy() 
        all_label_arr = self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])
        


        # if return_causal==True:
        #Creating the data-based on the main_model_mode
        if self.data_args["main_model_mode"]=="causal_removed_sp":
            print("Loading the data where the spurious-topic is removed")
            all_index_arr = all_index_arr_only_t0
        elif self.data_args["main_model_mode"]=="causal_rebalance_sp":
            print("Loading the data where the spurious-topic  is balanced (in causal mode)")
            #The change in p-value will already make the all_index_arr correct
            all_index_arr=all_index_arr
        elif self.data_args["main_model_mode"]=="causal_same_sp":
            print("Loading the data where the spurious-topic set to fixed val")
            all_index_arr=all_index_arr_t1_is_1
        
        #Correcting the p-value now
        if return_causal==True:
            self.data_args["topic_corr_list"][tidx1]=init_corr_value
        

        #Creating a dataframe if needed
        cat_dataset=None
        data_dict = dict(
                        #label=all_label_arr[:,self.data_args["main_topic"]+1],
                        label=all_label_arr[:,0],#topic_idx+1 for correct 
                        label_denoise = all_label_arr_nonoise[:,0],
                        input_idx = all_index_arr,
                        input_idx_t0_flip = all_index_arr_t0_flip,
                        input_idx_t1_flip = all_index_arr_t1_flip,
                        topic_label = all_label_arr[:,1:],
                        topic_label_denoise = all_label_arr_nonoise[:,1:],
                    )
        #Adding the counterfactual data if needed
        if return_cf==True:
            data_dict["input_idx_t0_cf"] = all_index_arr_cfactual_t0
            data_dict["input_idx_t1_cf"] = all_index_arr_cfactual_t1
        

        if return_fulldict==True:
            #Creating the dataframe from the input instead
            #This will be loaded in memory -> might create problem when scaling
            cat_dataset= data_dict
        else:
            cat_dataset = tf.data.Dataset.from_tensor_slices(
                                                        data_dict
            )
            #Batching the dataset
            # print(self.data_args["batch_size"])
            cat_dataset = cat_dataset.batch(self.data_args["batch_size"])
        
        return cat_dataset
    
    def toy_nlp_dataset_handler3(self,return_causal=False,return_cf=False,return_fulldict=False,add_mouli_cad=False):
        '''
        This function will generate the third version of the toy data handle
        which will have a particular causal graph to generate data:
           w  --> Y (confounder observed)
           tm --> Y (causal topic/treatment)

           w  --> ts (spurious topic/treatment)


        add_mouli_cad   : this will add the counterfactual data (wrt to topic) with the same main task lable.
                            The main intent is to learn an invariant representation wrt to topic whcih 
                            can be use to find the causal effect based on Mouli's Theorem-1.
        '''
        #Generating the topic labels 
        tm_conf_topic_label = np.array([
                        [0,0],
                        [0,1],
                        [1,0],
                        [1,1]
        ]*(self.data_args["num_sample"]//4))
        assert self.data_args["num_sample"]%4==0

        #Generating the spurious topic labels
        sp_topic_pval = self.data_args["sp_topic_pval"]

        #Flipping the label for sp_pval fraction of point
        num_leave  = int(sp_topic_pval*self.data_args["num_sample"])
        num_flip = self.data_args["num_sample"]-num_leave 
        flip_mask = np.array([1]*num_flip + [0]*num_leave)
        #Creating the spurious topic label
        sp_topic_label = (tm_conf_topic_label[:,1]+1*flip_mask)%2

        #Generating the main task labels
        #Getting the CPD of the Y=0 given topic causal and confounder
        y_cpd_tm_conf = defaultdict(dict)
        y_cpd_tm_conf[0][0]= 0.99 #0.9 
        y_cpd_tm_conf[0][1]= 0.30 #0.4
        y_cpd_tm_conf[1][0]= 0.70 #0.6
        y_cpd_tm_conf[1][1]= 0.01 #0.1

        #Generating the Y label
        y_label = []
        for yidx in range(self.data_args["num_sample"]):
            point_sample = np.random.uniform(0.0,1.0,1)
            tm   = tm_conf_topic_label[yidx,0]
            conf = tm_conf_topic_label[yidx,1]

            if point_sample<y_cpd_tm_conf[tm][conf]:
                y_label.append(0)
            else:
                y_label.append(1)
        y_label = np.array(y_label)

        causal_label=tm_conf_topic_label[:,0]
        confounder_label=tm_conf_topic_label[:,1]
        spurious_label=sp_topic_label
        
        #Creating the label array
        all_label_arr = np.stack([
                            y_label,
                            causal_label,
                            confounder_label,
                            spurious_label,
                        ],axis=1)

        #Adding noise to the labels
        self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])

        #Lets first measure the correlation between the topics
        label_corr_dict = self._print_label_correlation(all_label_arr)

        label_dict = dict(
                        topic_label=all_label_arr[:,1:],
                        causal=tm_conf_topic_label[:,0],
                        confounder=tm_conf_topic_label[:,1],
                        spurious=sp_topic_label,
                        y = y_label
        )

        #Creating the corresponding NLP example for this DAG
        print("Creating the dataset")
        all_example_widx_dict = self._create_nlp_dataset3_from_labels(label_dict)


        #Selecting the topic to keep as second
        if self.data_args["topic_name"]=="causal":
            topic_idx = 1 #1 :causal, 2: confounder 3: spurious
        elif self.data_args["topic_name"]=="confound":
            topic_idx = 2
        elif self.data_args["topic_name"]=="spurious":
            topic_idx = 3
        else:
            raise NotImplementedError()

        #Creating the dataframe
        data_dict = dict(
                        label=y_label,
                        label_denoise=y_label,
                        input_idx = all_example_widx_dict["all_example_widx_list"],
                        topic_label = all_label_arr[:,[topic_idx]],
                        topic_label_denoise = all_label_arr[:,[topic_idx]],
                        tcf_label = all_label_arr[:,2]
        )
        #Adding the counterfactual if required
        if return_cf==True:
            if topic_idx==1:
                data_dict["input_idx_t0_cf"] = all_example_widx_dict["all_example_cf_causal_widx_list"]
                data_dict["input_idx_t0_flip"] = all_example_widx_dict["all_example_cf_causal_widx_list"][:,0,:]
            elif topic_idx==2:
                data_dict["input_idx_t0_cf"] = all_example_widx_dict["all_example_cf_confound_widx_list"]
                data_dict["input_idx_t0_flip"] = all_example_widx_dict["all_example_cf_confound_widx_list"][:,0,:]
            elif topic_idx==3:
                data_dict["input_idx_t0_cf"] = all_example_widx_dict["all_example_cf_spurious_widx_list"]
                data_dict["input_idx_t0_flip"] = all_example_widx_dict["all_example_cf_spurious_widx_list"][:,0,:]

        #Next we will add the counterfactual in the main dataset if needed to do CAD for inv rep (Mouli for S1)
        #TODO: We can generalize the same code for all the dataset. Adapt later
        if add_mouli_cad==True: 
            #Adding the counterfactual label (this will be same as the main label cuz CAD for invariance)
            data_dict["label"] = np.concatenate([data_dict["label"],data_dict["label"]],axis=0)
            data_dict["label_denoise"] = np.concatenate([data_dict["label_denoise"],data_dict["label_denoise"]],axis=0)

            #Next we will add the input idx for the cf (which is same as the topic flip)
            init_input_idx = data_dict["input_idx"].copy()
            data_dict["input_idx"] = np.concatenate([data_dict["input_idx"],data_dict["input_idx_t0_cf"][:,0,:]],axis=0)
            #Adding the counterfactua / flip sentence for cf as input which is the input sentence for pdelta calc
            if return_cf==True:
                data_dict["input_idx_t0_cf"] = np.concatenate([data_dict["input_idx_t0_cf"],np.expand_dims(init_input_idx,axis=1)],axis=0)
                data_dict["input_idx_t0_flip"] = np.concatenate([data_dict["input_idx_t0_flip"],init_input_idx],axis=0)

            #Also we need to change the topic label to opposite since we added counterfactual
            #Assumption: The labels are 0/1 thus using 1-init_topic_label will flip things
            data_dict["topic_label"] = np.concatenate([data_dict["topic_label"],1-data_dict["topic_label"]],axis=0)
            data_dict["topic_label_denoise"] = np.concatenate([data_dict["topic_label_denoise"],1-data_dict["topic_label_denoise"]],axis=0)

            #Next we will add the confounder label
            #This will change if the debug tidx itself is confounder, other wise it will remain same
            if self.data_args["topic_name"]=="confound":
                data_dict["tcf_label"] = np.concatenate([data_dict["tcf_label"],1-data_dict["tcf_label"]],axis=0)
            else:
                data_dict["tcf_label"] = np.concatenate([data_dict["tcf_label"],data_dict["tcf_label"]],axis=0)
            
            #Next we need to reshuffle the dataset so that the counterfactual examples are evenly distributed
            shuffle_idx_list = list(range(data_dict["label"].shape[0]))
            np.random.shuffle(shuffle_idx_list)
            #Shuffling all the data with same shuffling index
            for dkey in data_dict.keys():
                data_dict[dkey] = data_dict[dkey][shuffle_idx_list]
        

        if self.data_args["return_label_dataset"]==True:
            raise NotImplementedError()
            #Creating the dataframe with only the labels based covariates instead of words
            data_dict["input_idx"] = all_label_arr[:,1:].astype("float32")
            #Creating the t0 counterfactual by flipping the t0 label
            t0_cf_X_arr = np.stack(
                            [(all_label_arr[:,1]+1)%2,all_label_arr[:,2],all_label_arr[:,3]],
                            axis=-1,
            )
            data_dict["input_idx_t0_cf"] = np.expand_dims(t0_cf_X_arr,axis=1).astype("float32")
            #Creating the t2 counterfactual by flipping the t2 label
            t1_cf_X_arr = np.stack(
                            [all_label_arr[:,1],all_label_arr[:,2],(all_label_arr[:,3]+1)%2],
                            axis=-1,
            )
            data_dict["input_idx_t1_cf"] = np.expand_dims(t1_cf_X_arr,axis=1).astype("float32")

        cat_dataset=None
        if return_fulldict==True:
            # cat_dataset=data_dict
            cat_dataset = (label_dict,label_corr_dict,all_example_widx_dict)
            return cat_dataset
        else:
            cat_dataset = tf.data.Dataset.from_tensor_slices(data_dict)
            cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

        return cat_dataset,label_corr_dict
    
    def _create_nlp_dataset3_from_labels(self,label_dict):
        '''
        '''
        print("Loading the embedding!")
        #Loading the gensim embedidng model
        self._load_full_gensim_word_embedding()

        #Generating the confounder topic model
        confound_topic_pos = [
                "good","best","awesome","teriffic","mighty","gigantic",
                "tremendous", "mega", "colossal",
        ]
        confound_topic_neg = [
                "bad", "inferior", "substandard", "inadequate", "rotten",
                "pathetic", "faulty", "defective",
        ]

        #Generating the causal topic model
        causal_topic_pos = [
                "rose","jasmine","tulip","lotus","daisy","sunflower","flower",
                "marigold","dahlia","orchid"
        ]
        causal_topic_neg = [
                "apple","mango","tomato","cherry","pear","fruit",
                "banana", "pear", "grapes"
        ]

        #Generating the spurious topic model
        spurious_topic_pos = [
                "comedy","romance","fantasy","sports","epic","animated",
                "adventure","science"
        ]
        spurious_topic_neg = [
                "horror","gore","crime","thriller","mystery","gangster",
                "drama","dark"
        ]

        #How many words do we allow here per topic
        num_words_in_topic = 20

        causal_topic_pos = causal_topic_pos[0:num_words_in_topic]
        causal_topic_neg = causal_topic_neg[0:num_words_in_topic]

        confound_topic_pos = confound_topic_pos[0:num_words_in_topic]
        confound_topic_neg = confound_topic_neg[0:num_words_in_topic]

        spurious_topic_pos = spurious_topic_pos[0:num_words_in_topic]
        spurious_topic_neg = spurious_topic_neg[0:num_words_in_topic]

        #Creating the sentences keepers
        all_example_sentence_list=[]
        all_example_notreat_dict={
                            "causal":[],
                            "confound":[],
                            "spurious":[]
        }#Contains the fragment of sentence without treated topic
        all_example_cf_dict = {
                            "causal":[],
                            "spurious":[],
                            "confound":[],
        }#This will contain the counterfactual example wrt to the topic
        for eidx in range(self.data_args["num_sample"]):
            causal_label   = label_dict["causal"][eidx]
            confound_label = label_dict["confounder"][eidx]
            spurious_label = label_dict["spurious"][eidx]

            sentence = None
            causal_fragment = None 
            confound_fragment = None

            #Getting the cf_fragment for each one
            causal_cf_fragment = None 
            confound_cf_fragment = None  

            #Getting one word from each list
            num_words_per_topic = 3
            causal_pos_frag = " " + " ".join(np.random.choice(causal_topic_pos,num_words_per_topic,replace=True).tolist()) + " "
            causal_neg_frag = " " + " ".join(np.random.choice(causal_topic_neg,num_words_per_topic,replace=True).tolist()) + " "

            confound_pos_frag = " " + " ".join(np.random.choice(confound_topic_pos,num_words_per_topic,replace=True).tolist()) + " "
            confound_neg_frag = " " + " ".join(np.random.choice(confound_topic_neg,num_words_per_topic,replace=True).tolist()) + " "

            spurious_pos_frag = " " + " ".join(np.random.choice(spurious_topic_pos,num_words_per_topic,replace=True).tolist()) + " "
            spurious_neg_frag = " " + " ".join(np.random.choice(spurious_topic_neg,num_words_per_topic,replace=True).tolist()) + " "

            #Dont keep the semantic meaning. Use explicit word for causal  fragment
            causal_fragment = None
            causal_cf_fragment = None 
            if causal_label==0:
                causal_fragment = causal_neg_frag
                causal_cf_fragment = causal_pos_frag
            else:
                causal_fragment = causal_pos_frag
                causal_cf_fragment = causal_neg_frag
            

            confound_fragment = None 
            confound_cf_fragment = None 
            if confound_label==0:
                confound_fragment = confound_neg_frag
                confound_cf_fragment = confound_pos_frag
            else:
                confound_fragment = confound_pos_frag
                confound_cf_fragment = confound_neg_frag

            #Here we will inject the confoundedness with certain probability
            point_sample = np.random.uniform(0.0,1.0,1)
            if point_sample<self.data_args["degree_confoundedness"]:
                confound_fragment = " "
                confound_cf_fragment = " "
            

            # Next adding the spurious tokens
            spurious_fragment = None 
            spurious_cf_fragment = None
            if spurious_label==0:
                spurious_fragment = spurious_neg_frag
                spurious_cf_fragment = spurious_pos_frag
            else:
                spurious_fragment = spurious_pos_frag
                spurious_cf_fragment = spurious_neg_frag


            #Adding the sentence to our keeper
            all_example_sentence_list.append(causal_fragment+confound_fragment+spurious_fragment)

            #Getting the no treatment example
            all_example_notreat_dict["causal"].append(confound_fragment+spurious_fragment)
            all_example_notreat_dict["confound"].append(causal_fragment+spurious_fragment)
            all_example_notreat_dict["spurious"].append(causal_fragment+confound_fragment)

            #Getting the counterfactual examples
            all_example_cf_dict["causal"].append(causal_cf_fragment+confound_fragment+spurious_fragment)
            all_example_cf_dict["spurious"].append(causal_fragment+confound_fragment+spurious_cf_fragment)
            all_example_cf_dict["confound"].append(causal_fragment+confound_cf_fragment+spurious_fragment)

        # pdb.set_trace()
        #Next we will have to conver all the sentence to thier widx 
        all_example_widx_list = self._convert_text_to_widx(all_example_sentence_list)
        #Converting the notreat sentence to widx
        all_example_notreat_causal_widx_list = self._convert_text_to_widx(all_example_notreat_dict["causal"])
        all_example_notreat_confound_widx_list = self._convert_text_to_widx(all_example_notreat_dict["confound"])
        all_example_notreat_spurious_widx_list = self._convert_text_to_widx(all_example_notreat_dict["spurious"])
        #Converting the cf sentence to widx
        all_example_cf_causal_widx_list = np.expand_dims(self._convert_text_to_widx(all_example_cf_dict["causal"]),axis=1)
        all_example_cf_spurious_widx_list = np.expand_dims(self._convert_text_to_widx(all_example_cf_dict["spurious"]),axis=1)
        all_example_cf_confound_widx_list = np.expand_dims(self._convert_text_to_widx(all_example_cf_dict["confound"]),axis=1)

        all_example_widx_dict = dict(
                        all_example_widx_list = all_example_widx_list,
                        all_example_notreat_causal_widx_list = all_example_notreat_causal_widx_list,
                        all_example_notreat_confound_widx_list = all_example_notreat_confound_widx_list,
                        all_example_notreat_spurious_widx_list = all_example_notreat_spurious_widx_list,
                        all_example_cf_causal_widx_list = all_example_cf_causal_widx_list,
                        all_example_cf_spurious_widx_list = all_example_cf_spurious_widx_list,
                        all_example_cf_confound_widx_list = all_example_cf_confound_widx_list,
        )

        return all_example_widx_dict
    
    def _generate_topic0_constituents(self,number_words,non_number_words,
                                    pos_label_list,neg_label_list,
                                    topic0_corr):
        '''
        '''
        tidx0 = 0
        point_sample = np.random.uniform(0.0,1.0,1)
        tpos_word = np.random.choice(number_words,10,replace=True).tolist()
        tneg_word = np.random.choice(non_number_words,10,replace=True).tolist()
        if point_sample<=topic0_corr:
            pos_add_topic0 = " " +  " ".join(tpos_word)+ " "
            neg_add_topic0 = " " +  " ".join(tneg_word)+ " "

            pos_label_list.append(1)
            neg_label_list.append(0)
        else:
            neg_add_topic0 = " " + " ".join(tpos_word)+ " "
            pos_add_topic0 = " " + " ".join(tneg_word)+ " "

            pos_label_list.append(0)
            neg_label_list.append(1)
        
        return pos_add_topic0,neg_add_topic0,pos_label_list,neg_label_list
    
    def _generate_topic0_cf_constituents(self,number_words,non_number_words,
                                        curr_topic0_label):
        '''
        '''      
        if curr_topic0_label==1:#this is positive class
            #Then we will have to send back the opposite class for this topic (neg)
            tneg_word = np.random.choice(non_number_words,10,replace=True).tolist()
            neg_add_topic0 = " " +  " ".join(tneg_word)+ " "

            return neg_add_topic0
        else:
            tpos_word = np.random.choice(number_words,10,replace=True).tolist()
            pos_add_topic0 = " " +  " ".join(tpos_word)+ " "

            return pos_add_topic0
        
    def _generate_topic1_constituents(self,pos_label_list,neg_label_list,
                                        topic1_corr):
        '''
        '''
        tidx1 = 1
        #Taking a differnet sample for this topic
        point_sample = np.random.uniform(0.0,1.0,1)
        if point_sample<=topic1_corr:
            pos_add_topic1 = " ".join(["fill"]*10)
            neg_add_topic1 = " "

            pos_label_list.append(1)
            neg_label_list.append(0)
        else:
            neg_add_topic1 = " ".join(["fill"]*10)
            pos_add_topic1 = " "

            pos_label_list.append(0)
            neg_label_list.append(1)

        return pos_add_topic1,neg_add_topic1,pos_label_list,neg_label_list
    
    def _generate_topic1_cf_constituents(self,curr_topic1_label):
        '''
        '''
        if curr_topic1_label==1:
            neg_add_topic1 = " "

            return neg_add_topic1
        else:
            pos_add_topic1 = " ".join(["fill"]*10)

            return pos_add_topic1

    def _generate_toy2_counterfactual(self,number_words,non_number_words,
                                            sentence_prefix,
                                            add_topic0,
                                            add_topic1,
                                            label_list,
                                            ):
        '''
        Generating the counterfactuals keeping the one feature same at a time
        '''
        cfactual_list_topic0=[] #Here the topic0 is transfored only (pos for tpoic0 inv)
        cfactual_list_topic1=[]

        topic0_label_idx=1
        topic1_label_idx=2


        for cidx in range(self.data_args["cfactuals_bsize"]):
            #Generating the counterfactual wrt to topic0
            # add_topic0_cf,_,_,_= self._generate_topic0_constituents(number_words=number_words,
            #                                     non_number_words=non_number_words,
            #                                     pos_label_list=[],
            #                                     neg_label_list=[],
            #                                     topic0_corr=0.5,#see both the variation
            # )
            #Generating using the opposite class of the topic method
            add_topic0_cf = self._generate_topic0_cf_constituents(number_words=number_words,
                                                non_number_words=non_number_words,
                                                curr_topic0_label=label_list[topic0_label_idx],#label of topic0
            )
            #Creating the counterfactual example
            cfactual_list_topic0.append(sentence_prefix+add_topic0_cf+add_topic1)
        


            #Next creating the counterfactual wrt to the topic1
            # add_topic1_cf,_,_,_ = self._generate_topic1_constituents(
            #                                     pos_label_list=[],
            #                                     neg_label_list=[],
            #                                     topic1_corr=0.5
            # )
            #Generating cf using the opposite class of the topic method
            add_topic1_cf = self._generate_topic1_cf_constituents(
                                                curr_topic1_label=label_list[topic1_label_idx],#label of topic1
            )
            #Creating the counterfactual example
            cfactual_list_topic1.append(sentence_prefix+add_topic0+add_topic1_cf)
        
        return cfactual_list_topic0,cfactual_list_topic1

    def toy_tabular_dataset_handler2(self,):
        '''
        This function will generate the tabular data in the same firmat as the
        toy_nlp_dataset_handler2, where we have:

        y --> X1
          --> X2
        And  on eatch edge we have the predictive correlation probability.

        The advantage of this tabular data is that, we know extaly what dimensions 
        correspond to the the causal/invariant and the spurious feature.
        So for the linear layer setup we can measure the w_sp/w_inv.
        '''
        mean_scale = 10.0
        sigma_lbound = 0.0
        sigma_ubound = self.data_args["tab_sigma_ubound"]

        #Creating the mean vectors for the spurious and inv dist
        mu_0 = np.random.randn(self.data_args["inv_dims"])*mean_scale
        mu_1 = mu_0.copy()
        assert self.data_args["inv_dims"]==self.data_args["sp_dims"],"Diff dims for feature"
        #Assuming we have digonal covariance matrix
        sigma_0 = np.random.uniform(low=sigma_lbound,
                                    high=sigma_ubound,
                                    size=self.data_args["inv_dims"]
        )
        sigma_1 = sigma_0.copy()


        #Creating the examples
        all_example_list = []
        all_example_list_t0_flip = []
        all_example_list_t1_flip = []   #These have the flipped corresponding topic
        all_example_list_only_t0 = []   #These example have spurious feature absent
        all_label_list= []
        for sidx in range(self.data_args["num_sample"]):
            pos_label_list = [1,]
            neg_label_list = [0,]

            #Creating the positive example
            #we wont have any causal fearture now which is fully predictive
            pos_example = []#"this is a positive template " #add feature here
            neg_example = []#"this is a negative example "


            #Creating the topics 1
            tidx0 = 0
            point_sample = np.random.uniform(0.0,1.0,1)
            tpos_word =  mu_0     + np.random.randn(self.data_args["inv_dims"])*sigma_0
            tneg_word = (-1*mu_0) + np.random.randn(self.data_args["inv_dims"])*sigma_0
            if point_sample<=self.data_args["topic_corr_list"][tidx0]:
                pos_add_topic0 = tpos_word.tolist()
                neg_add_topic0 = tneg_word.tolist()

                pos_label_list.append(1)
                neg_label_list.append(0)
            else:
                neg_add_topic0 = tpos_word.tolist()
                pos_add_topic0 = tneg_word.tolist()

                pos_label_list.append(0)
                neg_label_list.append(1)
            

            #Creating the topic 2
            tidx1 = 1
            #Taking a differnet sample for this topic
            point_sample = np.random.uniform(0.0,1.0,1)
            tpos_word =  mu_1     + np.random.randn(self.data_args["inv_dims"])*sigma_1
            tneg_word = (-1*mu_1) + np.random.randn(self.data_args["inv_dims"])*sigma_1
            if point_sample<=self.data_args["topic_corr_list"][tidx1]:
                pos_add_topic1 = tpos_word.tolist()
                neg_add_topic1 = tneg_word.tolist()

                pos_label_list.append(1)
                neg_label_list.append(0)
            else:
                neg_add_topic1 = tpos_word.tolist()
                pos_add_topic1 = tneg_word.tolist()

                pos_label_list.append(0)
                neg_label_list.append(1)

            
            #Constructing the examples
            pos_example_main = pos_example + pos_add_topic0 + pos_add_topic1 
            neg_example_main = neg_example + neg_add_topic0 + neg_add_topic1 
            all_example_list+=[pos_example_main,neg_example_main]

            #Constructing the examples by flipping topic0
            pos_example_t0 = pos_example + neg_add_topic0 + pos_add_topic1 
            neg_example_t0 = neg_example + pos_add_topic0 + neg_add_topic1 
            all_example_list_t0_flip+=[pos_example_t0,neg_example_t0]

            #Constructing the examples by flipping topic1
            pos_example_t1 = pos_example + pos_add_topic0 + neg_add_topic1 
            neg_example_t1 = neg_example + neg_add_topic0 + pos_add_topic1 
            all_example_list_t1_flip+=[pos_example_t1,neg_example_t1]

            #Creating the example which just have a topic0
            pos_example_only_t0 = pos_example + pos_add_topic0
            neg_example_only_t0 = neg_example + neg_add_topic0
            all_example_list_only_t0+=[pos_example_only_t0,neg_example_only_t0]

            #Creating example with only t1
            # pos_example_only_t1 = pos_example + pos_add_topic1
            # neg_example_only_t1 = neg_example + neg_add_topic1

            #Adding the example and lable
            all_label_list+=[pos_label_list,neg_label_list]


        #Converting the text example to input_idx 
        all_index_arr = np.array(all_example_list,np.float32)
        #Getting the input idx where feature0 is flipped (on-->off or off-->on)
        all_index_arr_t0_flip = np.array(all_example_list_t0_flip,np.float32)
        #Getting the input idx where feature1 is flipped (on-->off or off-->on)
        all_index_arr_t1_flip = np.array(all_example_list_t1_flip,np.float32)
        #Getting the input idx where only feature0 is present
        all_index_arr_only_t0 = np.array(all_example_list_only_t0,np.float32)

        
        #Creating the dataset object
        all_label_arr = np.array(all_label_list,np.int32)
        self._print_label_distribution(all_label_arr)
        #Shuffling the dataser (no need right now they are balanced)
        #Adding noise to the labels to have non-fully predictive causal features
        all_label_arr = self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])
        
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    #label=all_label_arr[:,self.data_args["main_topic"]+1],
                                    label=all_label_arr[:,0],
                                    input_idx = all_index_arr,
                                    input_idx_t0_flip = all_index_arr_t0_flip,
                                    input_idx_t1_flip = all_index_arr_t1_flip,
                                    input_idx_only_t0 = all_index_arr_only_t0,
                                    topic_label = all_label_arr[:,1:]
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

        return cat_dataset
    
    def _print_label_distribution(self,all_label_arr):
        '''
        To be sure that individual topics are even distributed in both the labels
        '''
        for tidx in range(self.data_args["num_topics"]):
            num_pos = np.sum(all_label_arr[:,tidx]==1)
            num_neg = np.sum(all_label_arr[:,tidx]==0)
            print("topic:{}\tnum_pos:{}\tnum_neg:{}".format(tidx,num_pos,num_neg))
        
        return
    
    def _add_noise_to_labels(self,all_label_arr,noise_ratio,only_main=False):
        '''
        Here we will add noise to each of the labels (main and or topic)
        '''
        for tidx in range(all_label_arr.shape[-1]):
            #Generating the flip mask
            num_flip = int(all_label_arr.shape[0]*noise_ratio)
            flip_idx = np.random.randint(all_label_arr.shape[0],size=num_flip).tolist()

            #Flipping the label
            all_label_arr[flip_idx,tidx] = np.logical_not(all_label_arr[flip_idx,tidx]==1)

            if only_main==True:
                print("WARNING: Adding the noise only in the main label, assuming that the main label is in the first index")
                break 
        
        #Printing the label correlation
        self._print_label_correlation(all_label_arr)

        return all_label_arr
    
    def _print_label_correlation(self,all_label_arr):
        '''
        '''
        #Calculating the topic correlation
        print("\n\n#############################################")
        print("Printing the label correlation")
        print("#############################################")
        label_corr_dict = defaultdict(dict)
        for iidx in range(all_label_arr.shape[-1]):
            #Getting the correlation with next topics
            for jidx in range(iidx+1,all_label_arr.shape[-1]):
                #Calcuating the correlation bw iidx and jidx
                corr = np.sum(all_label_arr[:,iidx]==all_label_arr[:,jidx])/(1.0*all_label_arr.shape[0])
                print("iidx:{}\tjidx:{}\tcorr:{:0.2f}".format(iidx,jidx,corr))
                label_corr_dict[iidx][jidx]=corr

        return label_corr_dict
    
    def _convert_text_to_widx(self,text_example_list):
        '''
        This function converts the input example in the text form to the word index
        based on the gensim key to index for use in the NBOW model.
        '''
        all_index_list = []
        #Converting the example to token idx
        for eidx,example in enumerate(text_example_list):
            #Splitting and tokenizing
            token_list = example.split()
            #Tokenizing the example
            index_list = [
                self.emb_model.key_to_index[token] 
                    for token in token_list
                        if token in self.emb_model.key_to_index
            ]

            #Padding the examples to the fixed length
            padding = [self.emb_model.key_to_index["unk"]]*(self.data_args["max_len"]-len(index_list))
            index_list = index_list + padding
            all_index_list.append(index_list[0:self.data_args["max_len"]])
        
        all_index_arr = np.array(all_index_list,np.int32)
        
        return all_index_arr

    def _tnlp_load_template_dataset(self,task_name):
        #Loading all the dataset for the given task
        task_path = self.data_args["path"]+task_name+"/{}.tsv"
        fnames = ["train","dev","test","train_other"]

        #Aggregating all the tmeplates
        all_example_df = []
        for fname in fnames:
            fpath = task_path.format(fname)
            fdf = pd.read_csv(fpath,sep="\t",names=["label","template"])
            #Filtering out the positive and negative labels
            fdf = fdf[(fdf["label"]==1) | (fdf["label"]==-1)]
            #Now making the negative lable as 0
            fdf.loc[fdf["label"]==-1,"label"]=0

            #Adding to the aggregate list
            all_example_df.append(fdf)
        
        template_df = pd.concat(all_example_df,ignore_index=True)

        return template_df

    def _tnlp_get_category_dataset(self,cat_name,
                                        causal_ratio,spuriousness_ratio,
                                        template_df):
        '''
        This will use the template dataset and induce bias towards a particular 
        group/demographies wrt to the task

        causal_ratio         : the accruacy we hope to achieve if we just use
                                causal feature relavant to the task (keep 0.75)
        spuriousness_ratio   : the accuracy which an indomain optimal classifier
                                will be able to use if they use these spurious feature.
                                (kepp 0.90)
        '''
        print("Creating category df for: {}".format(cat_name))
        #Getting the replacement dict
        replacement_dict = self._tnlp_catwise_replacement_dict_manual()[cat_name]

        #Now we will have to create instance of examples from the template dataset
        cat_df = template_df.copy(deep=True)
        num_examples = cat_df.shape[0]

        #Creating the flip label for causal limit
        num_keep_causal = int(causal_ratio*num_examples)
        causal_flip = np.random.permutation(
                    ["same"]*num_keep_causal+\
                    ["flip"]*(num_examples-num_keep_causal)
        )

        #Creating the flip label for the spuroius limit
        num_keep_spurious = int(spuriousness_ratio*num_examples)
        spurious_flip = np.random.permutation(
                    ["same"]*num_keep_spurious+\
                    ["flip"]*(num_examples-num_keep_spurious)
        )
        #Our flipper function
        def flip_label(label,flip_decision):
            if flip_decision=="flip":
                return 1 if label==0 else 0
            else:
                return label

        #Creating the cateogory dataframe
        label_list=[]
        doc_list = []
        pdoc_list = []
        for ridx in range(cat_df.shape[0]):
            #Flipping the label t have a causal threshold
            label = cat_df.iloc[ridx]["label"]
            label = flip_label(label,causal_flip[ridx])

            #Getting the text to replace the template with
            replace_frag = np.random.choice(replacement_dict[label],size=1)[0]
            #Again replacing #Adding noise to label wrt to spurious feature
            label = flip_label(label,spurious_flip[ridx])
            label_list.append(label)

            #Replacing the template
            template_text = cat_df.iloc[ridx]["template"]
            replaced_text = re.sub('XYZ',replace_frag,template_text)
            doc_list.append(replaced_text)

            #Processing the doc
            doc_tokens,doclen,token_list = self._clean_the_document(replaced_text)
            pdoc_list.append(token_list)
        
        #Now creating the category dataframe
        cat_df = pd.DataFrame(
                    dict(
                        label = label_list,
                        doc = doc_list,
                        pdoc = pdoc_list
                    )
        )

        print(cat_df.head())
        print("Num of Positive lable: {}".format(sum(label_list)))
        print("Num of Negative label: {}".format(num_examples-sum(label_list)))
        print("Causal Upper Bound of performance: {}".format(causal_ratio))
        print("Spurious Upper bound of performance: {}".format(spuriousness_ratio))
        print("====================================")
        return cat_df

    def _tnlp_catwise_replacement_dict_manual(self,):
        '''
        This will give us the thing which we will replace the template XYZ with
        in the example for a given category/domain
        '''
        replacement_dict = {
            "gender":{
                1:[
                                "The man",
                                "adam",
                ],
                0:[
                                "The women",
                                "eve",
                ],
            },
            "race":{
                1:[
                                "The white person",
                                "john",
                ],
                0:[
                                "The black person",
                                "will smith",
                ],
            },
            "orientation":{
                1:[
                                "The straight person",
                ],
                0:[
                                "The gay person",
                ],
            }
        }
        
        return replacement_dict

    def controlled_cebab_dataset_handler(self,return_causal=False,return_cf=False,nbow_mode=False,topic_idx=0):
        '''
        This dataset has pairs of counterfactual data along with labels on the topic
        on which they are purturbed.

        Labelling Conventions:
        NA

        topic_idx       : this will be used to generated the topic specific dataset with different topic number
                            to be used for training the combined dataset at once.
        '''
        #Creating the dataset
        cf_merged_df = self._get_cebab_dataframe()
        self._get_cebab_data_stats(cf_merged_df)
        self._get_true_causal_effect_cebab(cf_merged_df)

        #Getting the pbalanced df
        if self.data_args["topic_pval"]!=float("inf"):
            cf_merged_df = self._get_pbalanced_cebab_dataframe(cf_merged_df)
            self._get_cebab_data_stats(cf_merged_df)
            self._get_true_causal_effect_cebab(cf_merged_df)

    
        #Getting the list of input and counterfactuals
        sentence_list = cf_merged_df["init_sentence"].tolist()
        cf_sentence_lol = cf_merged_df["cf_{}_sentence".format(self.data_args["cebab_topic_name"])].tolist()
        
        if nbow_mode==False:
            #Tokenizing the input and getting ready for training
            input_idx,attn_mask,token_type_idx,cf_input_idx,cf_attn_mask,cf_token_type_idx\
                        = self._encode_sentences_for_te_cebab_transformer(sentence_list,cf_sentence_lol)
        else:
            attn_mask,cf_attn_mask=None,None
            input_idx,cf_input_idx = self._encode_sentences_for_te_cebab_nbow(sentence_list,cf_sentence_lol)

        #Creating the labels for the task
        y_label = cf_merged_df["init_sentiment_label"].tolist()
        topic_label = cf_merged_df["init_{}_label".format(self.data_args["cebab_topic_name"])].tolist()
        #We will repeat the topic label multiple times to help the combined training run smoothly
        all_label_arr = np.stack(
                                [y_label,]+[topic_label,]*(topic_idx+1),
                                axis=-1,
        )
        #Adding noise to the labels
        if self.data_args["noise_ratio"]!=None:
            all_label_arr = self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])

        #Creating the dataset dict
        data_dict=dict(
                        label=all_label_arr[:,0],
                        label_denoise=all_label_arr[:,0],
                        input_idx=input_idx,
                        attn_mask=attn_mask,
                        topic_label=all_label_arr[:,1:],
                        topic_label_denoise=all_label_arr[:,1:],
        )
        #Adding the topic numbered counterfactual dataset (used for indexing correctly)
        data_dict["input_idx_t{}_flip".format(topic_idx)]=cf_input_idx[:,0,:]
        data_dict["attn_mask_t{}_flip".format(topic_idx)]=cf_attn_mask[:,0,:]

        if return_cf==True:
            data_dict["input_idx_t{}_cf".format(topic_idx)]=cf_input_idx
            data_dict["attn_mask_t{}_cf".format(topic_idx)]=cf_attn_mask
        #Creating the dataset object ready for consumption
        cat_dataset=tf.data.Dataset.from_tensor_slices(data_dict)
        cat_dataset=cat_dataset.batch(self.data_args["batch_size"])
        # pdb.set_trace()

        return cat_dataset
    
    def _get_pbalanced_cebab_dataframe(self,cf_merged_df):
        '''
        '''
        #Shuffle the dataframe first
        cf_merged_df = cf_merged_df.sample(frac=1).reset_index(drop=True)
        #Getting the group mask
        grp1_mask,grp2_mask,grp3_mask,grp4_mask=self._get_cebab_group_mask(cf_merged_df)

        #Now getting the number of elements in the minoroty group
        num_minority = cf_merged_df[grp2_mask|grp4_mask].shape[0]
        assert (1-self.data_args["topic_pval"])>0.01,"Very high correlation"
        num_majority = int((self.data_args["topic_pval"]/(1-self.data_args["topic_pval"]))*num_minority)

        #If the number of majority required to achieve that correaltion is large, we will have to subsample the minority
        max_majority_samples = cf_merged_df[grp1_mask|grp3_mask].shape[0]
        if num_majority>max_majority_samples:
            num_majority = max_majority_samples
            num_minority = int(((1-self.data_args["topic_pval"])/self.data_args["topic_pval"])*num_majority)

        #Now getting the slice for each group
        majority_df = cf_merged_df[grp1_mask|grp3_mask][0:num_majority]
        minority_df = cf_merged_df[grp2_mask|grp4_mask][0:num_minority]

        #Merging the df
        pbalanced_df = pd.concat([majority_df,minority_df]).sample(frac=1).reset_index(drop=True)
        print("Number of samples before rebalancing:",cf_merged_df.shape[0])
        print("Number of samples after rebalancing:",pbalanced_df.shape[0])
        self.data_args["{}_num_samples".format(self.data_args["cebab_topic_name"])]=pbalanced_df.shape[0]

        # pdb.set_trace()

        return pbalanced_df       
    
    def _get_true_causal_effect_cebab(self,cf_merged_df):
        '''
        Since we have the true counterfactual along with the changed sentiment label for each sentence
        we can calculate the true causal effect of the topic. The possible problem is after changing the topic
        there might not be a majority vote on the sentiment. Maybe just discard them!
        '''
        ate_list = []
        for eidx in range(cf_merged_df.shape[0]):
            #Getting the main sentence things
            main_sentiment_label = cf_merged_df.iloc[eidx]["init_sentiment_label"]
            main_topic_label     = cf_merged_df.iloc[eidx]["init_{}_label".format(self.data_args["cebab_topic_name"])]

            #Now we will go through each of the counteractual sentence and get the ITE
            num_cf = len(cf_merged_df.iloc[eidx]["cf_{}_label".format(self.data_args["cebab_topic_name"])])
            for cidx in range(num_cf):
                cf_sentiment_label = cf_merged_df.iloc[eidx]["cf_{}_sentiment_label".format(self.data_args["cebab_topic_name"])][cidx]
                cf_topic_label     = cf_merged_df.iloc[eidx]["cf_{}_label".format(self.data_args["cebab_topic_name"])][cidx]

                #Skip if the topic is not changed
                if main_topic_label==cf_topic_label:
                    continue 
                elif main_topic_label==1:
                    ite = main_sentiment_label-cf_sentiment_label
                else:
                    ite = cf_sentiment_label - main_sentiment_label
                
                #Adding the ite to ate_list for averaging later
                ate_list.append(ite)
        
        print("ATE for {}: ".format(self.data_args["cebab_topic_name"]),np.mean(ate_list))
    
    def _encode_sentences_for_te_cebab_nbow(self,sentence_list,cf_sentence_lol):
        '''
        This will convert the cebab to word token to be processed using nbow based encoder 
        instead of the BERT based encoder.
        '''
        print("Loading the embedding!")
        #Loading the gensim embedidng model
        self._load_full_gensim_word_embedding()

        #Converting the input sentence into the word index
        input_index_arr = self._convert_text_to_widx(sentence_list)

        #Next converting the counterfactual examples to the word index
        cf_index_lol = []
        for cf_example_list in cf_sentence_lol:
            cf_index_lol.append(
                self._convert_text_to_widx(
                    cf_example_list[0:self.data_args["cfactuals_bsize"]]
                )
            )
        #Merging the index into one array -- (batch,num_cf,seq_length)
        cf_index_arr = np.stack(cf_index_lol,axis=0)
        
        return input_index_arr,cf_index_arr
    
    def _encode_sentences_for_te_cebab_transformer(self,sentence_list,cf_sentence_lol):
        '''
        This function will encode the sentence and their counterfactual for the
        treatment effect estimation for the BERT or RoBERTa based encoders
        '''
        #Now we will encode the input sentences
        encoded_doc = self.tokenizer(
                                sentence_list,
                                padding="max_length",
                                truncation=True,
                                max_length=self.data_args["max_len"],
                                return_tensors="tf"
        )
        input_idx = encoded_doc["input_ids"]
        attn_mask = encoded_doc["attention_mask"]
        token_type_idx = None 
        if "token_type_ids" in encoded_doc:
            token_type_idx = encoded_doc["token_type_ids"]
        


        #Next encoding the counterfactual sentences
        cf_input_idx_list = []
        cf_attn_mask_list = []
        cf_token_type_idx_list = []
        for cf_sentence_list  in cf_sentence_lol:
            cf_encoded_doc = self.tokenizer(
                                    cf_sentence_list[0:self.data_args["cfactuals_bsize"]],
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
            cf_input_idx = cf_encoded_doc["input_ids"]
            cf_attn_mask = cf_encoded_doc["attention_mask"]
            cf_token_type_idx = None 
            if "token_type_ids" in cf_encoded_doc:
                cf_token_type_idx = cf_encoded_doc["token_type_ids"]
            
            #Adding them to the list
            cf_input_idx_list.append(cf_input_idx)
            cf_attn_mask_list.append(cf_attn_mask)
            cf_token_type_idx_list.append(cf_token_type_idx)
        
        #Merging all the cf_idxs
        cf_input_idx = tf.stack(cf_input_idx_list,axis=0)
        cf_attn_mask = tf.stack(cf_attn_mask_list,axis=0)
        if cf_token_type_idx_list[0]!=None:
            cf_token_type_idx = tf.stack(cf_token_type_idx_list,axis=0)
        # pdb.set_trace()
        

        return input_idx,attn_mask,token_type_idx,cf_input_idx,cf_attn_mask,cf_token_type_idx
    
    def _get_cebab_data_stats(self,cf_merged_df):
        '''
        This fucntion will give us the proportion of the samples in each subgroup
        and the p-value of the topic with the main task.
        '''
        #Printing the aggreagate metrics of this dataset
        print("positive sentiment ratio: ",cf_merged_df["init_sentiment_label"].mean())
        grp1_mask,grp2_mask,grp3_mask,grp4_mask = self._get_cebab_group_mask(cf_merged_df)

        print("Estimated p-value in the current dataset: ",
                    cf_merged_df[grp1_mask|grp3_mask].shape[0]/cf_merged_df.shape[0])
        
        return 
    
    def _get_cebab_group_mask(self,cf_merged_df):
        #Getting the correlation statistics between the sentiment and the topic label
        grp1_mask = (cf_merged_df["init_sentiment_label"]==1) & (cf_merged_df["init_{}_label".format(self.data_args["cebab_topic_name"])]==1)
        grp2_mask = (cf_merged_df["init_sentiment_label"]==0) & (cf_merged_df["init_{}_label".format(self.data_args["cebab_topic_name"])]==1)
        grp3_mask = (cf_merged_df["init_sentiment_label"]==0) & (cf_merged_df["init_{}_label".format(self.data_args["cebab_topic_name"])]==0)
        grp4_mask = (cf_merged_df["init_sentiment_label"]==1) & (cf_merged_df["init_{}_label".format(self.data_args["cebab_topic_name"])]==0)

        return grp1_mask,grp2_mask,grp3_mask,grp4_mask

    def _get_cebab_dataframe(self,):
        '''
        '''
        #Loading the dataset from the hugginface library
        from datasets import load_dataset
        cebab = load_dataset("CEBaB/CEBaB")

        print("Reading the data into dataframe!")
        #Collecting all the examples
        all_example_list=[]
        #Dumping the dataset to a dataframe st. we can do selection and processing
        for eidx in range(len(cebab["train_inclusive"])):
            example_dict = dict(
                id        = cebab["train_inclusive"][eidx]["original_id"],
                sentence  = cebab["train_inclusive"][eidx]["description"],
                sentiment = cebab["train_inclusive"][eidx]["review_majority"],
                cf_topic  = cebab["train_inclusive"][eidx]["edit_type"],
                food_label = cebab["train_inclusive"][eidx]["food_aspect_majority"],
                ambiance_label = cebab["train_inclusive"][eidx]["ambiance_aspect_majority"],
                service_label = cebab["train_inclusive"][eidx]["service_aspect_majority"],
                noise_label = cebab["train_inclusive"][eidx]["noise_aspect_majority"],
            )
            #Skipping no majority and unknonwn labels example
            #Changelog 1: We will allow unknonwn topic label now
            if  example_dict["{}_label".format(self.data_args["cebab_topic_name"])]=="no majority":
                continue
            

            all_example_list.append(example_dict)
        #Creating a dataframe 
        example_df = pd.DataFrame(all_example_list)
        print("Number of examples post first filtering: ",example_df.shape[0])
        # pdb.set_trace()


        print("Merging the example and their counterfactuals into one!")
        #Grouping together the dataset (x,tcf)
        cf_merged_example_list = []
        #Getting the number of availabel counterfacutal for each example
        cf_len_list = []
        for exid,edf in example_df.groupby("id"):
            #Getting the example_dict for this edf
            cf_example_dict = self._get_cebab_main_counterfacutal_pair_impl2(edf)
            if cf_example_dict==None:
                continue

            #Adding the example dict to list
            num_cf = len(cf_example_dict["cf_{}_label".format(self.data_args["cebab_topic_name"])])
            if num_cf!=0:
                cf_len_list.append(num_cf)
                cf_merged_example_list.append(cf_example_dict)
        print("Avg. number of counterfactual per example:",np.mean(cf_len_list))
        #Expanding the merged example list (by reversing the ocunterfactual and main)
        if self.data_args["symmetrize_main_cf"]==True:
            cf_merged_example_list = self._symmetrize_cebab_actual_cf_pairs(cf_merged_example_list)

        #Creating the new dataframe with the merged examples
        cf_merged_df = pd.DataFrame(cf_merged_example_list)
        print("Number of Example post merging: ",cf_merged_df.shape[0])

        #We have to rebalance the dataset with equal number of pos and negs
        num_samples_pg = self.data_args["num_sample"]//2
        positive_df = cf_merged_df[cf_merged_df["init_sentiment_label"]==1]
        negative_df = cf_merged_df[cf_merged_df["init_sentiment_label"]==0]
        print("Number of initial positive examples:",positive_df.shape[0])
        print("Number of initial negative examples:",negative_df.shape[0])
        assert positive_df.shape[0]>=num_samples_pg and negative_df.shape[0]>=num_samples_pg,"Examples Exhausted!"

        positive_df_slice = positive_df[0:num_samples_pg]
        negative_df_slice = negative_df[0:num_samples_pg]

        balanced_cf_merged_df = pd.concat([positive_df_slice,negative_df_slice]).sample(frac=1).reset_index(drop=True)

        return balanced_cf_merged_df
    
    def _symmetrize_cebab_actual_cf_pairs(self,cf_merged_example_list):
        '''
        Since we have main example --> mapped to counterfactual examples,
        we could do a reverse mapping and get the cf --> main example mapping. 
        '''
        print("Number of example pre symmetrizing: ",len(cf_merged_example_list))
        cf_topic = self.data_args["cebab_topic_name"]
        reverse_example_list = []
        #Going over all the examples
        for main_example in cf_merged_example_list:
            #Going over all the counterfactual examples
            for cfidx in range(len(main_example["cf_{}_sentiment_label".format(cf_topic)])):
                cf_example_dict = {
                            "init_sentiment_label" : main_example["cf_{}_sentiment_label".format(cf_topic)][cfidx],
                            "init_sentence" : main_example["cf_{}_sentence".format(cf_topic)][cfidx],
                            "init_{}_label".format(cf_topic) : main_example["cf_{}_label".format(cf_topic)][cfidx],
                            "cf_{}_sentiment_label".format(cf_topic) : [main_example["init_sentiment_label"]],
                            "cf_{}_sentence".format(cf_topic) : [main_example["init_sentence"]],
                            "cf_{}_label".format(cf_topic) : [main_example["init_{}_label".format(cf_topic)]],
                }
                #Adding the example to the list
                reverse_example_list.append(cf_example_dict)
        #Now we will add all the reversed example to the main list and send back to MARS
        cf_merged_example_list = cf_merged_example_list + reverse_example_list
        print("Number of example post symmetrizing:",len(cf_merged_example_list))

        return cf_merged_example_list
    
    def _get_cebab_main_counterfacutal_pair_impl2(self,edf):
        '''
        Our main goal is to increase the number of example we have for CEBAB,
        becuase we have ground thruth here, thus we could check the true
        performance of Stage 1. In other real dataset we will not have the 
        true effect.


        One thing we could do is to combine the UNK and the NEG label as topic 0,
        and then we could see the effect of saying good food on the sentiment. 
        This will possibly dilute the ATE, but there is nothing technically wrong.

        '''
        def convert_wordlabel_to_num(word_label):
            # print(word_label)
            if word_label=="Negative" or word_label=="unknown":
                if self.data_args["cebab_topic_name"]=="noise":
                    return 1
                else:
                    return 0
            elif word_label=="Positive":
                if self.data_args["cebab_topic_name"]=="noise":
                    return 0
                else:
                    return 1
            raise NotImplementedError()
        
        def convert_sentiment_range_to_binary(sentiment_sval):
            #Creating the sentiment label
            #TODO: Should we keep the label 3 on one of the side
            #could help increase the example but will also increase difficulty
            if sentiment_sval=="no majority" or sentiment_sval=="3":
                return None 
            elif int(sentiment_sval)>3:
                return 1
            elif int(sentiment_sval)<3:
                return 0
            else:
                raise NotImplementedError() 
        

        #In this group we should have a main setence (with none)
        if "None" not in edf["cf_topic"].tolist() or\
            self.data_args["cebab_topic_name"] not in edf["cf_topic"].tolist():
            return None
        # print("Main and Counterfactual example present!")
        

        #Initializing the main example
        cf_example_dict=defaultdict(list)

        #First of getting the main example from the dataframe
        main_idx = None
        for midx in range(edf.shape[0]):
            #If we get the main example then fill the main part of the example
            if edf.iloc[midx]["cf_topic"]=="None":
                #Keeping track to leave it the next time
                main_idx = midx 

                #Getting the sentiment label
                sentiment_label = convert_sentiment_range_to_binary(
                                            edf.iloc[midx]["sentiment"]
                )
                #If the main sentence label is no-maj or mid-senti then leave
                if sentiment_label==None:
                    return None
                cf_example_dict["init_sentiment_label"]=sentiment_label
                
                #Getting the sentince and the initial topic label
                cf_example_dict["init_sentence"]=edf.iloc[midx]["sentence"]
                cf_example_dict["init_{}_label".format(self.data_args["cebab_topic_name"])]\
                                = convert_wordlabel_to_num(
                                    edf.iloc[midx]["{}_label".format(self.data_args["cebab_topic_name"])]
                                )

                #We are done finding the main example: Now chose the cf based on its topic label
                break
        # print("Main example Detected!")
        
        #Next we will search for the counterfactual examples
        for cidx in range(edf.shape[0]):
            #Getting the counterfactual topic 
            cf_topic = edf.iloc[cidx]["cf_topic"]
            #Skip the main example again
            if cidx==main_idx:
                continue 
            #Skipping the example if the edit goal was different, i.e likely not cf wrt to req topic
            if cf_topic!=self.data_args["cebab_topic_name"]:
                continue 
            

            #Now we will create the counterfactual examples
            #Getting the counterfactual sentiment labels
            cf_sentiment_label = convert_sentiment_range_to_binary(
                                            edf.iloc[cidx]["sentiment"]
            )
            #Getting the topic label for the counterfactual
            cf_topic_label = convert_wordlabel_to_num(
                                edf.iloc[cidx]["{}_label".format(cf_topic)]
            )


            #Skip if the sentiment is non-maj or rated 3
            if cf_sentiment_label==None:
                continue
            #Skip if the topic label has not changed from the main
            if cf_topic_label==cf_example_dict["init_{}_label".format(cf_topic)]:
                continue 
            
            #Now we will add the sentence if its senti correct and cf is done
            cf_example_dict["cf_{}_sentiment_label".format(cf_topic)].append(cf_sentiment_label)
            cf_example_dict["cf_{}_sentence".format(cf_topic)].append(edf.iloc[cidx]["sentence"])
            cf_example_dict["cf_{}_label".format(cf_topic)].append(cf_topic_label)

        return cf_example_dict
    
    def get_cebab_sentiment_only_dataset(self,nbow_mode=False):
        '''
        Since we are wasting the samples if we only consider the topic specific daset to 
        train the main model. So we will now get the merged dataset to train the main base
        model and use the counterfactual data to train the alpha.

        Also, this will be the more natural setting since, we will have limited counterfactual 
        but sufficient task specific dataset.
        '''
        #Loading the dataset from the hugginface library
        from datasets import load_dataset
        cebab = load_dataset("CEBaB/CEBaB")

        print("Reading the data into dataframe!")
        #Collecting all the examples
        all_example_list=[]
        #Dumping the dataset to a dataframe st. we can do selection and processing
        for eidx in range(len(cebab["train_inclusive"])):
            #Lets subsample here itself wrt to the bad sentiment labels
            sentiment = cebab["train_inclusive"][eidx]["review_majority"]
            if sentiment=="no majority" or sentiment=="3":
                continue
            elif int(sentiment)>3:
                sentiment = 1
            elif int(sentiment)<3:
                sentiment = 0
            else:
                raise NotImplementedError()
            

            #Getting the only relavant field for us
            example_dict = dict(
                sentence  = cebab["train_inclusive"][eidx]["description"],
                sentiment = sentiment,
            )
            all_example_list.append(example_dict)
        
        #Creating a dataframe out of it first
        all_example_df = pd.DataFrame(all_example_list)
        positive_df = all_example_df[all_example_df["sentiment"]==1]
        negative_df = all_example_df[all_example_df["sentiment"]==0]
        num_keep = min(positive_df.shape[0],negative_df.shape[0])

        #Lets add a cap on the overall dataset size also
        print("Full CeBaB dataset size: ",all_example_df.shape[0])
        print("Num Positive: ",positive_df.shape[0])
        print("Num Negative: ",negative_df.shape[0])
        all_dataset_size = self.data_args["num_sample_cebab_all"]

        #Now we will create the index for these examples
        positive_df_slice = positive_df[0:num_keep][0:all_dataset_size//2]
        negative_df_slice = negative_df[0:num_keep][0:all_dataset_size//2]

        #Merging the dataframe
        balanced_all_example_df = pd.concat(
                        [positive_df_slice,negative_df_slice]
        ).sample(frac=1).reset_index(drop=True)
        
        #Now we will convert the examples to the tensorflow dataset
        sentence_list = balanced_all_example_df["sentence"].tolist()
        y_label = balanced_all_example_df["sentiment"].tolist()


        #We will user the BERT tokenizer if we will use transformer
        if nbow_mode==False:
            #Now we will encode the input sentences
            encoded_doc = self.tokenizer(
                                    sentence_list,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
            input_idx = encoded_doc["input_ids"]
            attn_mask = encoded_doc["attention_mask"]
            token_type_idx = None 
            if "token_type_ids" in encoded_doc:
                token_type_idx = encoded_doc["token_type_ids"]

            #Creating the dataset
            data_dict = dict(
                                label = y_label,
                                input_idx = input_idx,
                                attn_mask = attn_mask,
            )
        else:
            #We need to encode the sentence for the word vectors
            print("Loading the embedding!")
            #Loading the gensim embedidng model
            self._load_full_gensim_word_embedding()

            #Converting the input sentence into the word index
            input_index_arr = self._convert_text_to_widx(sentence_list)

            #Creating the dataset
            data_dict = dict(
                                label = y_label,
                                input_idx = input_idx,
            )
        
        #Adding the empty topic label array
        data_dict["topic_label"]=[None]*len(y_label)
        data_dict["input_idx_t0_cf"]=[None]*len(y_label)
        data_dict["attn_mask_t0_cf"]=[None]*len(y_label)
        
        #Batching the and sending the dataset
        cat_dataset=tf.data.Dataset.from_tensor_slices(data_dict)
        cat_dataset=cat_dataset.batch(self.data_args["batch_size"])

        return cat_dataset
    
    def controlled_cda_dataset_handler(self,dataset,return_causal=False,return_cf=False,nbow_mode=False):
        '''
        In this dataset we could create the manually generated counterfactuals. 
        The only problem is for all the axis/groups we will consider all of them are going to be non-causal
        so how can we check if our method is not biased in giving zero causal effect all the time.
        '''
        #Getting the pbalnaced dataframe
        if dataset=="civilcomments":
            pbalanced_df = self._get_civilcomment_dataframe()
        elif dataset=="aae":
            pbalanced_df = self._get_cda_aae_dataframe()
        else:
            raise NotImplementedError()
        #We dont have true counterfactual to get the true causal effect. 
        #TODO: Create semi synthetic dataset where we could do this so that we could verify the stage1


        #Now converting the dataframe into the tensor for the traning
        sentence_list = pbalanced_df["sentence"].tolist()
        cf_sentence_lol = pbalanced_df["cf_sentence"].tolist()
        #Tokenizing the input and getting ready for training
        input_idx,attn_mask,token_type_idx,cf_input_idx,cf_attn_mask,cf_token_type_idx\
                    = self._encode_sentences_for_te_cebab_transformer(sentence_list,cf_sentence_lol)

        #Creating the label
        y_label = pbalanced_df["label"].tolist()
        topic_label = pbalanced_df["topic_label"].tolist()

        #Creating the topic label 
        all_label_arr = np.stack(
                                [y_label,topic_label],
                                axis=-1,
        )
        #Adding the noise to the labels
        if self.data_args["noise_ratio"]!=None:
            all_label_arr = self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])

        #Creating the dataset dict
        data_dict=dict(
                        label=all_label_arr[:,0],
                        label_denoise=all_label_arr[:,0],
                        input_idx=input_idx,
                        attn_mask=attn_mask,
                        topic_label=all_label_arr[:,1:],
                        topic_label_denoise=all_label_arr[:,1:],
                        input_idx_t0_flip=cf_input_idx[:,0,:],
                        attn_mask_t0_flip=cf_attn_mask[:,0,:],
        )
        if return_cf==True:
            data_dict["input_idx_t0_cf"]=cf_input_idx
            data_dict["attn_mask_t0_cf"]=cf_attn_mask
        #Creating the dataset object ready for consumption
        cat_dataset=tf.data.Dataset.from_tensor_slices(data_dict)
        cat_dataset=cat_dataset.batch(self.data_args["batch_size"])
        # pdb.set_trace()

        return cat_dataset
    
    def _get_cda_aae_dataframe(self,):
        '''
        This will load the already preprocessed dataset which had the counterfactual added from the
        the gpt.
        '''
        #Loading the dataframe
        df_path = self.data_args["path"] + "/aae_cf_added_df.csv"
        aae_df = pd.read_csv(df_path,sep="\t")
        assert aae_df[aae_df["cf_sentence"]=="got exception for this cf. ERRORCODE404"].shape[0]==0,"cf_unavailable"
        self._get_civilcomment_df_stats(aae_df)

        #Next we could rebalance the dataframe using the same civilcomments function
        pbalanced_df = self._get_rebalanced_civilcomment_df(aae_df)

        return pbalanced_df

    def _get_civilcomment_dataframe(self,):
        '''
        '''
        #TODO: Add the hashing for the csv of preprocessed dataframe


        import tensorflow_datasets as tfds
        ds = tfds.load("civil_comments/CivilCommentsIdentities")

        #Getting the topic list ready to generate the counterfactual 
        topic_word_dict = self._get_civilcomment_topics()
        topic_wordmap = self._get_civilcomment_topic_wordmap()

        #Collecting all the example into one place
        all_example_list = []
        print("Reading the civilcomments example list")
        num_subsample = 100000 
        tbar = tqdm(range(len(ds["train"].take(num_subsample))))
        for tidx,ex_d in zip(tbar,ds["train"].take(num_subsample)):
            #Getting the topic label 
            topic_label = None
            # pdb.set_trace()
            '''
            Since we are looking at the counterfactual where we remove the topic we need not 
            have example where there is ony male and only female. We are temoving the the topic as
            counterfactual.
            '''
            if self.data_args["topic_name"]=="religion":
                if ex_d["christian"].numpy()>0.5 or ex_d["muslim"].numpy()>0.5:
                    topic_label = 1
                else:
                    topic_label = 0
            elif self.data_args["topic_name"]=="gender":
                if ex_d["male"].numpy()>0.5 or ex_d["female"].numpy()>0.5:
                    #TODO: if we use the exact 1.0 instead of 0.5 then what?
                    #Every main exmple will have the label
                    topic_label = 1
                else:
                    topic_label = 0
            elif self.data_args["topic_name"]=="race":
                if ex_d["black"].numpy()>0.5 or ex_d["white"].numpy()>0.5:
                    topic_label = 1
                else:
                    topic_label = 0
            elif self.data_args["topic_name"]=="toxic":
                #We will use the word list itself to get the topic label
                raise NotImplementedError()
            else:
                raise NotImplementedError()


            #Creating the example
            sentence = ex_d["text"].numpy().decode("utf-8").lower()
            example_dict = dict(
                            sentence = sentence,
                            label = 1 if ex_d["toxicity"]>0.5 else 0,
                            topic_label = topic_label,
                            cf_sentence = [
                                self._get_civilcommment_counterfactual(
                                                    sentence,
                                                    topic_word_dict,
                                                    topic_wordmap,
                                                    self.data_args["replace_strategy"],
                                ),
                            ],
            )
            all_example_list.append(example_dict)
        
        #Creating the dataframe
        all_example_df = pd.DataFrame(all_example_list)
        self._get_civilcomment_df_stats(all_example_df)
        # pdb.set_trace()

        #Lets rebalance the dataset
        pbalanced_df = self._get_rebalanced_civilcomment_df(all_example_df)
        
        
        return pbalanced_df
    
    def _get_rebalanced_civilcomment_df(self,example_df):
        '''
        '''
        #Getting the inividual group mask
        grp1_mask,grp2_mask,grp3_mask,grp4_mask = self._get_civilcomment_df_group_mask(example_df)
        min_grp_val = min(
                    example_df[grp1_mask].shape[0],
                    example_df[grp2_mask].shape[0],
                    example_df[grp3_mask].shape[0],
                    example_df[grp4_mask].shape[0]
        )

        #Now we will subsample the dataset to create artifical correlation
        req_sample_pg = self.data_args["num_sample"]//4
        assert req_sample_pg<=min_grp_val,"Examples exhausted!"

        #Now we will rebalance the rest of the sub-groups
        majority_grp_df = pd.concat([
                                example_df[grp1_mask][0:req_sample_pg],
                                example_df[grp3_mask][0:req_sample_pg],
        ])
        #Getting the minuroty subgroup individual group sample 
        num_mino_sub_pg = int(req_sample_pg*(1-self.data_args["topic_pval"])/self.data_args["topic_pval"])
        minority_grp_df = pd.concat([
                                example_df[grp2_mask][0:num_mino_sub_pg],
                                example_df[grp4_mask][0:num_mino_sub_pg],
        ])

        pbalanced_df = pd.concat([majority_grp_df,minority_grp_df]).sample(frac=1).reset_index(drop=True)
        #Now getting the stats of the dataframe
        self._get_civilcomment_df_stats(pbalanced_df)

        return pbalanced_df
    
    def _get_civilcomment_df_stats(self,example_df):
        '''
        '''
        grp1_mask,grp2_mask,grp3_mask,grp4_mask = self._get_civilcomment_df_group_mask(example_df)

        print("Number of element in grp1: ",example_df[grp1_mask].shape[0])
        print("Number of element in grp2: ",example_df[grp2_mask].shape[0])
        print("Number of element in grp3: ",example_df[grp3_mask].shape[0])
        print("Number of element in grp4: ",example_df[grp4_mask].shape[0])
        print("pvalue: ",example_df[grp1_mask | grp3_mask].shape[0]/example_df.shape[0])

        #Getting the number of positive and negative samples ration
        positive_df = example_df[example_df["label"]==1]
        negative_df = example_df[example_df["label"]==0]
        print("Number of actual positive examples: ",positive_df.shape[0])
        print("Number of actual negative examples: ",negative_df.shape[0])

        return 
    
    def _get_civilcomment_df_group_mask(self,example_df):
        '''
        '''
        grp1_mask = ((example_df["topic_label"]==1)  & (example_df["label"]==1))
        grp2_mask = ((example_df["topic_label"]==1)  & (example_df["label"]==0))
        grp3_mask = ((example_df["topic_label"]==0)  & (example_df["label"]==0))
        grp4_mask = ((example_df["topic_label"]==0)  & (example_df["label"]==1))

        return grp1_mask,grp2_mask,grp3_mask,grp4_mask

    def _get_civilcommment_counterfactual(self,sentence,topic_word_dict,topic_wordmap,replace_strategy):
        '''
        Here we will change the sentence using words based topic model to generate counterfactuals
        We will now follow two strategy to generente the counterfactual.

        Strategy 1: Replace the words from the given topic from the sentnece. 
                    This has potential problem that, sometime the topic specific mention are
                    important for labelling the sentence. 
        
        Strategy 2: Here we will map the individual topic word to the counterfactual words within
                    that topic itself. This will not change the sentence structure then. 
                    And is a more natural version to say the toxicity should not depend on one type
                    of religion!
        '''
        sentence_tokens = sentence.split(" ")
        sentence_cf_tokens = []

        #Here we will do the removal interventions instead of replacing them with male version
        if replace_strategy=="remove":
            for word in sentence_tokens:
                if word not in topic_word_dict:
                    sentence_cf_tokens.append(word)

        elif replace_strategy=="map_replace":
            #Here we will replace the sentence with word the mapping words we have
            for word in sentence_tokens:
                if word not in topic_wordmap:
                    sentence_cf_tokens.append(word)
                else:
                    sentence_cf_tokens.append(topic_wordmap[word])
        else:
            raise NotImplementedError()
        

        #Creating the counterfactual sentence
        sentence_cf = " ".join(sentence_cf_tokens)
        return sentence_cf
    
    def _get_civilcomment_topics(self,):
        '''
        '''
        #Creating the race topic
        race_topic = [
                "black", "white", "supremacists", "obama", "clinton", "trump", "john",
                "negro", "blacks", "whites", "nigga",
                #These are more ethnicity related terms
                #"asian", "ethnicity", "aloha", "india","pakistan","iran",
                #"mexican", "mexico", "british", 
        ]

        #Creating the gneder topic
        gender_topic = [
                "male","patriarchy","feminist","woman","men","man","female","women","he","she",
                "him","his", "mother", "father", "feminists" ,"boy", "girl"
        ]

        #Creating the religion topic
        religion_topic = [
                "christian","muslim","allah","jesus","john","mohammed", "catholic","priest",
                "church", "islamic", "islam", "hijab"
        ]

        #Creating the toxic topic : this should have greater causal effect
        toxic_topic = [
                "hatred", "pronto", "crooked", "f**k", "fuck", "di*ks", "dicks",
                "stupid", "pathetic", "bigot", "bigot", "worthless", "shit",
                "rascal", "bastard", "whackjob", "whack", "crazy"
        ]

        if self.data_args["topic_name"]=="gender":
            topic_word_list = gender_topic
        elif self.data_args["topic_name"]=="race":
            topic_word_list = race_topic
        elif self.data_args["topic_name"]=="religion":
            topic_word_list = religion_topic
        elif self.data_args["topic_name"]=="toxic":
            topic_word_list = toxic_topic

        #Extending the topic set
        if(self.data_args["extend_topic_set"]==True):
            self._load_word_embedding_for_nbr_extn()
            topic_word_list = list(self._extend_topic_set_wordembedding(topic_word_list))
            print("topic_word_list:\n",topic_word_list)
        
        #Sorting such that the largest words are in front so that we dont replace the small subword 
        #(will leave signal in the sentence then)
        topic_word_list.sort(key=lambda s:len(s),reverse=True)

        #TODO: Is sorted listing is maintained when we create the dict? 
        topic_word_dict = {word:True for word in topic_word_list}
        
        return topic_word_dict

    def _get_civilcomment_topic_wordmap(self,):
        '''
        This will give us the mapping of indicidual words in the topic to create a more natural
        counterfatual where we dont remove things, but flip the topic label.
        '''
        def create_bidict(wordmap):
            wordmap_cp = wordmap.copy()
            for key,val in wordmap.items():
                assert val not in wordmap_cp,"Value already exists in the wordmap"
                wordmap_cp[val]=key
            return wordmap_cp 

        #Creating the wordmap for religion words (later we could expand this using word analogy)
        religion_wordmap = {
                                "christian" : "muslim",
                                "jesus"     : "allah",
                                "john"      : "mohammed",
                                "catholic"  : "sunni",
                                "priest"    : "imam",
                                "church"    : "mosque",
        }
        religion_wordmap = create_bidict(religion_wordmap)

        #Creating the word map for the gender
        gender_wordmap = {
                                "male"      : "female",
                                "patriarchy": "matriarchy",
                                "patriarch" : "matriarch",
                                "feminist"  : "chauvinist",
                                "man"       : "woman",
                                "men"       : "women",
                                "he"        : "she",
                                "him"       : "his",
                                "father"    : "mother",
                                "boy"       : "girl",
                                "uncle"     : "aunt",

        }
        gender_wordmap = create_bidict(gender_wordmap)

        #Creating the race wordmap
        race_wordmap = {
                                "white"     : "black",
                                "trump"     : "obama",
                                "master"    : "slave",

        }
        race_wordmap = create_bidict(race_wordmap)

        #Creating the toxic word map
        toxic_wordmap = {}

        if self.data_args["topic_name"]=="gender":
            topic_wordmap = gender_wordmap
        elif self.data_args["topic_name"]=="race":
            topic_wordmap = race_wordmap
        elif self.data_args["topic_name"]=="religion":
            topic_wordmap = religion_wordmap
        elif self.data_args["topic_name"]=="toxic":
            topic_wordmap = toxic_wordmap
        
        return topic_wordmap        

    def controlled_multinli_dataset_handler(self,return_causal=False):
        '''
        Here we will create a multi-nli dataset where we simplify the setting
        to make it close to what we were doing in the toyNLP2.

        Instead of three-way classification, we will just have two classes
            1. Contradiction
            2. Non-Contradiction
        
        And the spurious topic will be:
            1. Negation 
            2. Non-negation words

        We will synthetically control the correlation in the dataset to get the variation
        across different value of p-value. One of the setting of p-value could be the
        natural one where we dont make any synthetic balancing of the topic labels
        to have a particular p-value.
        '''
        #Getting the multinli_dataframe
        example_df = self._get_multinli_dataframe()
        # pdb.set_trace()

        #Switching off the causal data generation arg
        if self.data_args["main_model_mode"]=="causal_same_sp" \
                and return_causal==True:
            pbalanced_df = self._get_no_neg_dataframe(
                                                example_df=example_df,
                                                group_ulim=41000,
            )
        elif self.data_args["main_model_mode"]=="causal_rebalance_sp" \
                and return_causal==True:
            #Changing the correlation value
            init_corr_value = self.data_args["neg_topic_corr"]
            self.data_args["neg_topic_corr"]=0.5

            #Getting the balanced dataset
            pbalanced_df = self._get_pbalanced_dataframe(
                                                    example_df=example_df,
                                                    group_ulim=41000,
            )
            #Resetting the initial p-value (upde will happenin in case of causal return)
            self.data_args["neg_topic_corr"]=init_corr_value
        else:
            pbalanced_df = self._get_pbalanced_dataframe(
                                                    example_df=example_df,
                                                    group_ulim=41000,
            )

        #Getting the labels array
        all_label_arr = np.stack([
                                pbalanced_df["main_label"].to_numpy(),
                                pbalanced_df["neg_topic_label"].to_numpy(),
        ],axis=-1)
        self._print_label_distribution(all_label_arr)
        #Adding noise to the labels to have non-fully predictive causal features
        all_label_arr = self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])


        #Now encoding the text to be readable by model
        input_idx,attn_mask,token_type_idx,flip_input_idx,flip_attn_mask,flip_token_type_idx\
                                 = self._get_bert_tokenized_inputs(pbalanced_df)

        #Now we are ready to create our dataset object
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    input_idx=input_idx,
                                    attn_mask=attn_mask,
                                    input_idx_t0_flip=flip_input_idx,
                                    attn_mask_t0_flip=flip_attn_mask,
                                    label=all_label_arr[:,0],
                                    topic_label=all_label_arr[:,1:]
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])
        return cat_dataset
      
    def _get_multinli_dataframe(self,):
        '''
        Path Assumption:
        "dataset/multinli_1.0/"
            1. "multinli_1.0_train.jsonl"
            2. "multinli_1.0_dev.jsonl"


        Labelling Convention:
            1. Positive label = 1
            2. Negative Label = 0
        '''
        negation_topic_words = ["negation","no","never","nothing"]
        def get_multinli_negation_topic(sentence):
            '''
            This will label a sentence positive (+1) if it contains either of :
                1. nobody
                2. no
                3. never
                4. nothing
            '''
            for word in negation_topic_words:
                if word in sentence:
                    return 1
            return 0
        
        def get_negation_flipped_sentence(sentence,negation_label):
            '''
            Here we will operate in multiple modes:

            1. add_non_negation : here we will add negation words to the sentence
                                  which dont have negations. [half the data]
            2. remove_negation  : this will remove the negation words from the 
                                  sentence which contain the negation words
            3. replace_negation : this will replace the negation words with something else
                                  which has similar meaning. So it their embedding is very
                                  different from the negation words then it will change the
                                  decision. But this seems biased
            '''
            # sentence=sentence.copy()
            if negation_label==0:
                sentence = sentence + " " + np.random.choice(negation_topic_words)
            else:
                if self.data_args["neg1_flip_method"]=="remove_negation":
                    for word in negation_topic_words:
                        sentence = sentence.replace(word," ")
                elif self.data_args["neg1_flip_method"]=="replace_negation":
                    for word in negation_topic_words:
                        sentence = sentence.replace(word,"dont")
                else:
                    #So that we dont make a mistake of adding them
                    sentence = sentence
            
            return sentence
        
        def add_examples_from_file(fname,example_list):
            print("Reading file: {}".format(fname))
            with jsonlines.open(self.data_args["path"]+"/"+fname) as rhandle:
                for example_json in rhandle.iter():
                    example_dict = dict(
                                    sentence1 = example_json["sentence1"],
                                    sentence2 = example_json["sentence2"],
                                    main_label = 1 if example_json["gold_label"]=="contradiction" else 0,
                                    neg_topic_label = get_multinli_negation_topic(example_json["sentence2"])
                    )
                    #Flipping the spurious feature
                    flip_sentence2 = get_negation_flipped_sentence(
                                                sentence=example_dict["sentence2"],
                                                negation_label=example_dict["neg_topic_label"],
                    )
                    example_dict["flip_sentence2"]=flip_sentence2

                    example_list.append(example_dict)
            
            return example_list
        
        #Getting all the examples in the train and dev set 
        train_fname = "multinli_1.0_train.jsonl"
        valid_fname = "multinli_1.0_dev_matched.jsonl" #matched contains same generes as train
        example_list = []
        example_list = add_examples_from_file(train_fname,example_list)
        example_list = add_examples_from_file(valid_fname,example_list)

        #Merging all the data into one dataframe
        example_df = pd.DataFrame(example_list)
        
        return example_df
    
    def _get_pbalanced_dataframe(self,example_df,group_ulim):
        '''
        Assumption is that both the main labels and the topic labels are binary right now
        '''
        #The maximum possible example in a group (decided by inspection)
        assert self.data_args["num_sample"]<=2*group_ulim,"Examples Exhausted"

        #Shuffling the dataframe first
        example_df = example_df.sample(frac=1).reset_index(drop=True)

        #Assigning the number of example in each group (group1=group3 and group2=group4)
        num_group1 = int((self.data_args["num_sample"]/2.0)*self.data_args["neg_topic_corr"])
        num_group4 = int((self.data_args["num_sample"]/2.0)*(1-self.data_args["neg_topic_corr"]))

        #Sampling the dataset for group1 (m=+,t=+) and group4 (m=+,t=-)
        group1_df = example_df[
                                (example_df["main_label"]==1) & (example_df["neg_topic_label"]==1)
                    ][0:num_group1]
        group4_df = example_df[
                                (example_df["main_label"]==1) & (example_df["neg_topic_label"]==0)
                    ][0:num_group4]
        
        #Sampling the dataset for group3(m=-,t=-) and group2 (m=-,t=+)
        group3_df = example_df[
                                (example_df["main_label"]==0) & (example_df["neg_topic_label"]==0)
                    ][0:num_group1]
        group2_df = example_df[
                                (example_df["main_label"]==0) & (example_df["neg_topic_label"]==1)
                    ][0:num_group4]


        #Concatenating the overall dataset
        pbalanced_df = pd.concat([group1_df,group4_df,group3_df,group2_df]).sample(frac=1).reset_index(drop=True)
        return pbalanced_df
    
    def _get_no_neg_dataframe(self,example_df,group_ulim):
        '''
        '''
        #The maximum possible example in a group (decided by inspection)
        assert self.data_args["num_sample"]<=2*group_ulim,"Examples Exhausted"

        #Shuffling the dataframe first
        example_df = example_df.sample(frac=1).reset_index(drop=True)

        num_positive = int(self.data_args["num_sample"]//2)
        #Filtering out only those example which dont have negation
        positive_df = example_df[
            (example_df["neg_topic_label"]==0) & (example_df["main_label"]==1)
        ][0:num_positive]

        negative_df = example_df[
            (example_df["neg_topic_label"]==0) & (example_df["main_label"]==0)
        ][0:num_positive]

        pbalanced_df = pd.concat([positive_df,negative_df]).sample(frac=1).reset_index(drop=True)
        return pbalanced_df
    
    def _get_bert_tokenized_inputs(self,pbalanced_df):
        '''
        '''
        #Getting the list of sentence 1 and sentence2 from the df
        sentence1_list, sentence2_list, flip_sentence2_list = [],[],[]
        for eidx in range(pbalanced_df.shape[0]):
            sentence1_list.append(pbalanced_df.iloc[eidx]["sentence1"])
            sentence2_list.append(pbalanced_df.iloc[eidx]["sentence2"])
            flip_sentence2_list.append(pbalanced_df.iloc[eidx]["flip_sentence2"])
        

        #Now we will tokenize using our pre-trained tokenizer
        encoded_doc = self.tokenizer(
                                    sentence1_list,#[CLS] will be added upfront automatically
                                    sentence2_list, #will automatically add the seperater [SEP]
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
        input_idx = encoded_doc["input_ids"]
        attn_mask = encoded_doc["attention_mask"]
        token_type_idx=None
        if "token_type_ids" in encoded_doc:
            token_type_idx = encoded_doc["token_type_ids"]


        #Tokenizing the flip inputs
        flip_encoded_doc = self.tokenizer(
                                    sentence1_list,#[CLS] will be added upfront automatically
                                    flip_sentence2_list, #will automatically add the seperater [SEP]
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
        flip_input_idx = flip_encoded_doc["input_ids"]
        flip_attn_mask = flip_encoded_doc["attention_mask"]
        flip_token_type_idx = None
        if "token_type_ids" in flip_encoded_doc:
            flip_token_type_idx = flip_encoded_doc["token_type_ids"]


        return input_idx,attn_mask,token_type_idx,flip_input_idx,flip_attn_mask,flip_token_type_idx
    
    def controlled_twitter_dataset_handler(self,return_causal=False):
        '''
        This function will add the twitter dataset in the pipeline
        as the 2nd NLP dataset where we want to experiment.


        AAE:
        Task: Sentiment
        Spurious Attribute: Race
        Here we dont know how to do the pdelta_calculation becuase
        we dont know about how race information looks in the input.

        Maybe that's a good thing to see the results in the dataset
        where we cant intervene. But then we dont have any way to know
        if the results are correact except the Acc(Smin) metric.


        PAN16:
        Task: Mention
        Spuriosu Attribute: Gender
        This dataset contains the tweets with mention task and gender 
        and age is also labelled along with it. 
        Here we could easily pertub the gender from the dataset
        '''
        #Getting the example dataframe
        example_df = self._get_twitter_dataframe()


        #Switching off the causal data generation arg
        if self.data_args["main_model_mode"]=="causal_same_sp" \
                and return_causal==True:
            pbalanced_df = self._get_no_neg_dataframe(
                                                example_df=example_df,
                                                group_ulim=80000,
            )
        elif self.data_args["main_model_mode"]=="causal_rebalance_sp" \
                and return_causal==True:
            #Changing the correlation value
            init_corr_value = self.data_args["neg_topic_corr"]
            self.data_args["neg_topic_corr"]=0.5

            #Getting the balanced dataset
            pbalanced_df = self._get_pbalanced_dataframe(
                                                    example_df=example_df,
                                                    group_ulim=80000,
            )
            #Resetting the initial p-value (upde will happenin in case of causal return)
            self.data_args["neg_topic_corr"]=init_corr_value
        else:
            pbalanced_df = self._get_pbalanced_dataframe(
                                                    example_df=example_df,
                                                    group_ulim=80000,
            )


        #Getting the labels array
        all_label_arr = np.stack([
                                pbalanced_df["main_label"].to_numpy(),
                                pbalanced_df["neg_topic_label"].to_numpy(),
        ],axis=-1)
        self._print_label_distribution(all_label_arr)
        #Adding noise to the labels to have non-fully predictive causal features
        all_label_arr = self._add_noise_to_labels(all_label_arr,self.data_args["noise_ratio"])

        #Now encoding the text to be readable by model
        input_idx,attn_mask,token_type_idx,flip_input_idx,flip_attn_mask,flip_token_type_idx\
                                    = self._get_bert_tokenized_inputs_twitter(pbalanced_df)

        #Now we are ready to create our dataset object
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    input_idx=input_idx,
                                    attn_mask=attn_mask,
                                    input_idx_t0_flip=flip_input_idx,
                                    attn_mask_t0_flip=flip_attn_mask,
                                    label=all_label_arr[:,0],
                                    topic_label=all_label_arr[:,1:]
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])
        return cat_dataset
    
    def _get_twitter_dataframe(self,):
        '''
        '''
        male_topic_words = [
            "his","he","him","himself","man","male","actor",
            "husband","father","uncle",
            "gentleman","masculine","chap","bloke","lad","dude",
            "bro","brother","son"
        ]
        female_topic_words = [ 
            "her","she","herself","women","actress","lady","girl",
            "gal","sister","daughter","wife","mother","aunt","nanny"
        ]

        #Generating the flip
        def get_flipped_sentence(sentence):
            '''
            '''
            if "pan16" in self.data_args["path"]:
                #Now generating the flip for the sentence
                word_list=[]
                for word in sentence.split():
                    #If we have a male related word we will replace with female word
                    if word in male_topic_words:
                        if self.data_args["neg1_flip_method"]=="remove_negation":
                            continue
                        elif self.data_args["neg1_flip_method"]=="replace_negation":
                            #Sampling a female related word
                            female_word = str(
                                    np.random.choice(female_topic_words,1)[0]
                            )
                            word_list.append(female_word)
                        else:
                            raise NotImplementedError()
                    elif word in female_topic_words:
                        if self.data_args["neg1_flip_method"]=="remove_negation":
                            continue 
                        elif self.data_args["neg1_flip_method"]=="replace_negation":
                            #Sampling a male related words
                            male_word = str(
                                    np.random.choice(male_topic_words,1)[0]
                            )
                            word_list.append(male_word)
                        else:
                            raise NotImplementedError()
                    else:
                        word_list.append(word)
                sentence = " ".join(word_list)
                return sentence
            elif "aae" in self.data_args["path"]:
                #Currently we dont know how to make pertubation to the race
                #Think
                return sentence
            else:
                raise NotImplementedError()
        
        def print_topic_model_topic_correlation(example_dict_list,topic_label):
            '''
            This function will see if we are correctly capturing the topic
            in our dataset by measuring the correlation bw male and female topic
            with the topic label
            '''
            if "aae" in self.data_args["path"]:
                return

            #This is only for the gender data we are doing
            male_counter = 0.0
            female_counter = 0.0
            for example in example_dict_list:
                sentence = example["sentence"]
                if(example["neg_topic_label"]!=topic_label):
                    continue
                for word in sentence.split():
                    if word in male_topic_words:
                        male_counter +=1
                    elif word in female_topic_words:
                        female_counter+=1
            
            print("topic label:{:}\tmale_ratio:{:0.2f}\tfemale_ratio{:0.2f}".format(
                               topic_label,
                               male_counter/(male_counter+female_counter),
                               female_counter/(male_counter+female_counter)
            ))


        #First of all we have to read the files and put them in single place
        def get_example_from_file(fname,main_label,topic_label):
            example_dict_list =[]
            with open(self.data_args["path"]+"/"+fname,"r",encoding="ISO-8859-1") as rhandle:
                for eidx,example in enumerate(rhandle):
                    #Nowmalizing the sentence
                    example = example.strip().lower()
                    # print("Reading the example:{}".format(example))


                    #Creating the example dict
                    example_dict_list.append(
                        dict(
                            sentence = example,
                            main_label=main_label,
                            neg_topic_label=topic_label,#just using same name as multinlp
                            flip_sentence=get_flipped_sentence(example)
                        )
                    )
            return example_dict_list
        

        #Getting all the examples
        all_example_dict_list=[]
        all_example_dict_list+=get_example_from_file(
                            fname="pos_pos",
                            main_label=1,
                            topic_label=1
        )
        all_example_dict_list+=get_example_from_file(
                            fname="pos_neg",
                            main_label=1,
                            topic_label=0
        )
        all_example_dict_list+=get_example_from_file(
                            fname="neg_pos",
                            main_label=0,
                            topic_label=1
        )
        all_example_dict_list+=get_example_from_file(
                            fname="neg_neg",
                            main_label=0,
                            topic_label=0
        )

        #Printing the label correlation
        print("\n\nPrinting the topic-label Correlation with topic model")
        print_topic_model_topic_correlation(all_example_dict_list,1)
        print_topic_model_topic_correlation(all_example_dict_list,0)
        print("\n\n")


        #Creating the dataframe
        example_df = pd.DataFrame(all_example_dict_list)
        # pdb.set_trace()
        return example_df
    
    def _get_bert_tokenized_inputs_twitter(self,pbalanced_df):
        '''
        '''
        #Getting the list of sentence 1 and sentence2 from the df
        sentence_list, flip_sentence_list = [],[]
        for eidx in range(pbalanced_df.shape[0]):
            sentence_list.append(pbalanced_df.iloc[eidx]["sentence"])
            flip_sentence_list.append(pbalanced_df.iloc[eidx]["flip_sentence"])
        
        #Tokenize the sentence with our pre-trained tokenizer
        encoded_doc = self.tokenizer(
                                    sentence_list,#[CLS] will be added upfront automatically
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
        input_idx = encoded_doc["input_ids"]
        attn_mask = encoded_doc["attention_mask"]
        token_type_idx=None
        if "token_type_ids" in encoded_doc:
            token_type_idx = encoded_doc["token_type_ids"]

        #Tokenizing the flip sentence
        flip_encoded_doc = self.tokenizer(
                                    flip_sentence_list, #will automatically add the seperater [SEP]
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.data_args["max_len"],
                                    return_tensors="tf"
            )
        flip_input_idx = flip_encoded_doc["input_ids"]
        flip_attn_mask = flip_encoded_doc["attention_mask"]
        flip_token_type_idx=None
        if "token_type_ids" in flip_encoded_doc:
            flip_token_type_idx = flip_encoded_doc["token_type_ids"]

        return input_idx,attn_mask,token_type_idx,flip_input_idx,flip_attn_mask,flip_token_type_idx


if __name__=="__main__":
    #Creating the data handler
    # data_args={}
    # data_args["max_len"]=100        
    # data_args["emb_path"]="random"
    # data_args["train_split"]=0.8
    # data_args["epsilon"] = 1e-3         #for numerical stability in sample weights
    # data_handle = DataHandler(data_args)

    # #Now creating our dataset from domain1 (original sentiment)
    # # domain1_path = "counterfactually-augmented-data-master/sentiment/orig/"
    # # data_handle.data_handler_ltdiff_paper_sentiment(domain1_path)

    # #Getting the dataset from amaon reviews 
    # cat_list = ["beauty","software","appliance","faishon","giftcard","magazine"]
    # path = "dataset/amazon/"
    # data_handle.data_handler_amazon_reviews(path,cat_list,1000)
    # pdb.set_trace()


    #Testing the data-handler
    data_args={}
    data_args["path"]="dataset/multinli_1.0/"
    data_args["transformer_name"]="roberta-base"
    data_args["num_sample"]=1100
    data_args["neg_topic_corr"]=0.7
    data_args["batch_size"]=32
    data_args["max_len"]=200
    data_args["num_topics"]=1
    data_args["noise_ratio"]=0.0
    data_args["run_num"]=14
    data_args["neg1_flip_method"]="remove_negation"
    data_handler = DataHandleTransformer(data_args)
    # cat_dataset=data_handler.controlled_multinli_dataset_handler()
    data_args["cebab_topic_name"]="food"
    data_args["cfactuals_bsize"]=1
    data_handler.controlled_cebab_dataset_handler()
    # data_args["topic_name"]="toxic"
    # data_args["extend_topic_set"]=True 
    # data_args["num_neigh"]=20
    # data_args["emb_path"]="glove-wiki-gigaword-100"
    # data_args["vocab_path"]="assets/word2vec_10000_200d_labels.tsv"
    # data_args["topic_pval"]=0.9
    # data_args["replace_strategy"]="map_replace"
    # data_handler.controlled_civilcomments_dataset_handler()
    # pdb.set_trace()


    #Testing the twitter datahandler
    # data_args={}
    # # data_args["path"]="dataset/twitter_pan16_mention_gender"
    # data_args["path"]="dataset/twitter_aae_sentiment_race"
    # data_args["transformer_name"]="roberta-base"
    # data_args["num_sample"]=1000
    # data_args["neg_topic_corr"]=0.7
    # data_args["batch_size"]=100
    # data_args["max_len"]=200
    # data_args["num_topics"]=1
    # data_args["noise_ratio"]=0.0
    # data_args["run_num"]=14
    # data_args["neg1_flip_method"]="remove_negation"

    # data_handler = DataHandleTransformer(data_args)
    # cat_dataset=data_handler.controlled_twitter_dataset_handler()
    # pdb.set_trace()




    
