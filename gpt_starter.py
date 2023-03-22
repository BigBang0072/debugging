import os

import openai
openai.api_key = "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"


start_phrase = '''
Convert the following sentence in the style of an African American speaker to Non-Hispanic white American speaker:


'''
    




def generate_question(sentence):
    response = openai.Completion.create(model="curie", #text-davinci-003", 
                                        prompt=start_phrase+sentence,
                                        max_tokens=100,
    )
    print(response)
    answer = response['choices'][0]['text'].replace(' .', '.').strip()
    return answer



if __name__=="__main__":
    response = generate_question(
                sentence="Friday is movie theater night and I need a movie buddy who wants to go ??????"
    )
    print(response)
    
