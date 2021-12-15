from collections import defaultdict
import numpy as np
import pandas as pd 
import tensorflow as tf
import gensim.downloader as gensim_api

import string
import spacy
nlp = spacy.load("en_core_web_lg",disable=["tagger", "parser", "lemmatizer", "ner", "textcat"])
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neighbors import NearestNeighbors

import os
import gzip
import json
import re 
import pprint
import pdb
pp=pprint.PrettyPrinter(indent=4)
import random

#Setting the random seed
random.seed(22)
np.random.seed(22)

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
        self.tokenizer = AutoTokenizer.from_pretrained(data_args["transformer_name"])

        self.delimiter=",|\?|\!|-|\*| |  |;|\.|\(|\)|\n|\"|:|'|/|&|`|[|]|\{|\}|\>|\<"
        self.word_count=defaultdict(int)
        self.filter_dict=None
    
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

        gender =set([ 
            "he", "she", "it", "male", "female", "john", "kennedy", "victoria", 
            "peter", "luna", "harry", "ron", "susan", "actor", "actress", "father",
            "mother", "nurse", "firefighter", "warrior", "man", "women", "god",
            "goddess"
        ])

        electronics = set([ 
            "phone", "computer", "circuit", "battery", "television", "remote",
            "headset", "charger", "telephone", "radio", "antenna", "tower", "signal",
            "screen", "keboard", "mouse", "keypad", "desktop", "fan", "ac", "cooler",
            "wire", "solder"
        ])

        pronoun = set([ 
            "he", "she", "it", "they", "them","i", "we", "themselves", "thy","thou",
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

        movies = set([ 
            "movie", "animation", "film", "show", "theatre", "hollywood",
            "disney", "producer", "director", "actor", "actress" 
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
            pos_adjective,neg_adjective,negations,adverbs,
            #religion,gender,electronics,pronoun,kitchen,genre,
            #arts,books,clothes,groceries,movies,pets,phone,tools,
            male,female,white,black,straight,gay,
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
            
            #Binary Feature
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
            topic_label=0
            if topic_feature[debug_topic_idx]>0:
                topic_label=1
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
        label_list = np.array(label_list)
        topic_label_list=np.array(topic_label_list)
        #Getting the main task class mask
        main_class0_mask = (label_list==0)
        main_class1_mask = (label_list==1)
        def get_topic_segmentation(class_mask,topic_label_arr,name):
            topic_label_class = topic_label_arr[class_mask]
            print("class:{}\tnum_topic_0:{}\tnum_topic_1:{}".format(
                                        name,
                                        topic_label_class.shape[0]-np.sum(topic_label_class),
                                        np.sum(topic_label_class)
            ))
        get_topic_segmentation(main_class0_mask,topic_label_list,"0")
        get_topic_segmentation(main_class1_mask,topic_label_list,"1")
        # pdb.set_trace()



        #Creating the dataset for this category
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=label_list,
                                    input_idx = np.array(all_index_list),
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
            topic_label=0
            if topic_feature[debug_topic_idx]>0:
                topic_label=1
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


        #Creating the dataset for this category
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=np.array(label_list),
                                    # topic=np.array(topic_list),
                                    # topic_weight=np.array(topic_weight_list),
                                    input_idx = input_idx,
                                    attn_mask = attn_mask,
                                    # topic_feature=topic_feature,
                                    topic_label = np.array(topic_label_list)
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

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
                                            pdoc_name="pdoc",
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
    
    def toy_nlp_dataset_handler2(self,):
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
        non_number_word = [
            "nice","device","try","picture","signature","trailer","harry","potter",
            "malfoy","john","switch","taste","glove","baloon"
        ]

        #Creating the examples
        all_example_list = []
        all_label_list= []
        for sidx in range(self.data_args["num_sample"]):
            pos_label_list = [1,]
            neg_label_list = [0,]

            #Creating the positive example
            pos_example = "this is a positive template "
            neg_example = "this is a negative example "


            #Creating the topics 1
            tidx0 = 0
            point_sample = np.random.uniform(0.0,1.0,1)
            tpos_word = np.random.choice(number_words,10,replace=True).tolist()
            tneg_word = np.random.choice(non_number_word,10,replace=True).tolist()
            if point_sample<=self.data_args["topic_corr_list"][tidx0]:
                pos_example += " ".join(tpos_word)+ " "
                neg_example += " ".join(tneg_word)+ " "

                pos_label_list.append(1)
                neg_label_list.append(0)
            else:
                neg_example += " ".join(tpos_word)
                pos_example += " ".join(tneg_word)

                pos_label_list.append(0)
                neg_label_list.append(1)
            

            #Creating the topic 2
            tidx1 = 1
            if point_sample<=self.data_args["topic_corr_list"][tidx1]:
                pos_example += " ".join(["fill"]*10)

                pos_label_list.append(1)
                neg_label_list.append(0)
            else:
                neg_example += " ".join(["fill"]*10)

                pos_label_list.append(0)
                neg_label_list.append(1)

            #Converting the examples to the token

            #Adding the example and lable
            all_example_list+=[pos_example,neg_example]
            all_label_list+=[pos_label_list,neg_label_list]


        all_index_list = []
        #Converting the example to token idx
        for eidx,example in enumerate(all_example_list):
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
            all_index_list.append(index_list)
        
        #Creating the dataset object
        all_index_arr = np.array(all_index_list,np.int32)
        all_label_arr = np.array(all_label_list,np.int32)
        #Shuffling the dataser (no need right now they are balanced)
        
        cat_dataset = tf.data.Dataset.from_tensor_slices(
                                dict(
                                    label=all_label_arr[:,0],
                                    input_idx = all_index_arr,
                                    topic_label = all_label_arr[:,1:]
                                )
        )

        #Batching the dataset
        cat_dataset = cat_dataset.batch(self.data_args["batch_size"])

        return cat_dataset

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






    
