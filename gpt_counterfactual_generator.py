import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pdb
import pandas as pd

import openai
openai.api_key = "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"


from nlp_data_handle import DataHandleTransformer

class CounterfactualGenerator():
    '''
    This class will be used to generate the automated counterfactuals using the OpenAI api. 
    '''
    def __init__(self,cf_args):
        '''
        '''
        self.cf_args=cf_args 

        #Creating the prompt for AAE dataset
        self.prompt_aa2nhw = '''
        Convert the following sentence in the style of African American speaker to Non-Hispanic white American speaker:

        '''
        self.prompt_nhw2aa = '''
        Convert the following sentence in the style of Non-Hispanic white American speaker to African American speaker:

        '''

    def generate_aae_counterfactual(self,):
        '''
        This function will generate the counterfactual for the aae dataset where we change the 
        style of the speaker from African American to Non-Hispoanic White English.
        '''
        #Getting the slice of corpus for which we want to generate the cf
        print("Getting the dataframe")
        aae_df = self._get_aae_dataframe()
        #Saving the dataframe in the 
        save_path = self.cf_args["df_save_path"]+"aae_df.csv"
        aae_df.to_csv(save_path,sep="\t")

        #Generating the counterfactual sentences
        cf_sentence_list = []
        print("Generating the counterfactual sentences")
        with open(self.cf_args["df_save_path"]+"cf_example_list.txt","w") as whandle:
            for eidx in range(aae_df.shape[0]):
                try:
                    sentence = aae_df.iloc[eidx]["sentence"]
                    topic_label = aae_df.iloc[eidx]["topic_label"] 
                    prompt=None
                    #topic=1 are the tweets from the aae race
                    if topic_label==1:
                        prompt = self.prompt_aa2nhw
                    else:
                        prompt = self.prompt_nhw2aa
                    #Getting the counterfactual
                    cf_sentence = self._get_sentence_counterfactual(
                                                                prompt=prompt,
                                                                sentence=sentence
                    )
                    cf_sentence_list.append(cf_sentence)
                    print("=========================================\n")
                    print("eidx:{}/{}\ntopic_label:{}\nsentence:{}\ncf_sentence:{}".format(
                                                                eidx,
                                                                aae_df.shape[0],
                                                                topic_label,
                                                                sentence,
                                                                cf_sentence
                    ))

                    #Writing the list of sentence in the file
                    whandle.write("=======================================\n")
                    whandle.write("sentence:{}\n".format(sentence))
                    whandle.write("topic_label:{}\n".format(topic_label))
                    whandle.write("cf_sentence:{}\n".format(cf_sentence))
                    print("DONE:written to file\n\n")
                except:
                    cf_sentence_list.append("got exception for this cf. ERRORCODE404")
                    continue

        #Adding the counterfactual to the dataframe
        aae_df = aae_df.assign(cf_sentence=cf_sentence_list)
        #Saving the dataframe in the disk with cf
        save_path = self.cf_args["df_save_path"]+"aae_cf_added_df.csv"
        aae_df.to_csv(save_path,sep="\t")
        #Reloading the df to test
        aae_df = pd.read_csv(save_path,sep="\t")

        print("Number of unfinished sentences:",aae_df[aae_df["cf_sentence"]=="got exception for this cf. ERRORCODE404"].shape[0])
        # pdb.set_trace()
        return aae_df
    
    def correct_unfilled_cf_sentences(self,):
        '''
        If for some reason the main dataframe were not filled correctly, this will refill them by running the API again.
        '''
        #Loading the dataframe
        save_path = self.cf_args["df_save_path"]+self.cf_args["cf_added_fname"]
        df = pd.read_csv(save_path,sep="\t")

        cf_sentence_list=[]
        for eidx in range(df.shape[0]):
            #Skipping the sentence which are already filled
            if df.iloc[eidx]["cf_sentence"]!="got exception for this cf. ERRORCODE404":
                cf_sentence_list.append(df.iloc[eidx]["cf_sentence"])
                continue
            
            try:
                sentence = df.iloc[eidx]["sentence"]
                topic_label = df.iloc[eidx]["topic_label"] 
                prompt=None
                #topic=1 are the tweets from the aae race
                if topic_label==1:
                    prompt = self.prompt_aa2nhw
                else:
                    prompt = self.prompt_nhw2aa
                #Getting the counterfactual
                cf_sentence = self._get_sentence_counterfactual(
                                                            prompt=prompt,
                                                            sentence=sentence
                )
                #Adding the counterfactual to the dataframe
                cf_sentence_list.append(cf_sentence) 

                #Logging the result
                print("=========================================\n")
                print("eidx:{}/{}\ntopic_label:{}\nsentence:{}\ncf_sentence:{}".format(
                                                            eidx,
                                                            df.shape[0],
                                                            topic_label,
                                                            sentence,
                                                            cf_sentence
                ))
            except:
                print("got exception for this cf. ERRORCODE404")
                continue
        
        #Adding the counterfactual to the dataframe
        df = df.assign(cf_sentence=cf_sentence_list)
        #Saving the dataframe in the disk with cf
        save_path = self.cf_args["df_save_path"]+self.cf_args["cf_added_fname"]
        df.to_csv(save_path,sep="\t")
        #Reloading the df to test
        df = pd.read_csv(save_path,sep="\t")

        #Testing exception again
        print("Number of unfinished sentences:",df[df["cf_sentence"]=="got exception for this cf. ERRORCODE404"].shape[0])
        return
        
    def _get_aae_dataframe(self,):
        '''

        '''
        data_args={}
        data_args["path"]=cf_args["path"]
        data_args["transformer_name"]="roberta-base"
        data_args["num_sample"]=self.cf_args["num_sample"]
        data_args["neg_topic_corr"]=0.5
        data_args["num_topics"]=1
        data_args["neg1_flip_method"]="remove_negation"
        data_args["run_num"]=0
        #Creating the data handler
        data_handler = DataHandleTransformer(data_args)

        #Getting the dataframe
        example_df = data_handler._get_twitter_dataframe()
        pbalanced_df = data_handler._get_pbalanced_dataframe(
                                        example_df=example_df,
                                        group_ulim=80000,
        )

        #Rebranding the dataframes in the same lingo as current dfs
        pbalanced_df.rename(
                    columns={
                            "sentence":"sentence",
                            "main_label":"label",
                            "neg_topic_label":"topic_label",
                            "flip_sentence":"old_word_based_cf_sentence"
                    },
                    inplace=True,
        )
        data_handler._get_civilcomment_df_stats(pbalanced_df)
        return pbalanced_df
    
    def _get_sentence_counterfactual(self,prompt,sentence):
        '''
        This function will generate the counterfactual sentence from the
        openai api.
        '''
        response = openai.Completion.create(
                                        model=cf_args["openai_model_name"],
                                        prompt=prompt+sentence,
                                        max_tokens=cf_args["max_tokens"],
        )
        cf_sentence = response['choices'][0]['text'].replace(' .', '.').strip()
        return cf_sentence

if __name__=="__main__":
    cf_args={}
    cf_args["path"]="dataset/twitter_aae_sentiment_race"
    cf_args["df_save_path"]="dataset/twitter_aae_sentiment_race/"
    cf_args["cf_added_fname"]="aae_cf_added_df.csv"
    cf_args["openai_model_name"]="text-davinci-003"  #text-davinci-003", 
    cf_args["max_tokens"]=100
    cf_args["num_sample"]=10000

    #Creating the counterfactual generator
    cf_generator = CounterfactualGenerator(cf_args)
    # cf_generator.generate_aae_counterfactual()
    cf_generator.correct_unfilled_cf_sentences()

        

        



