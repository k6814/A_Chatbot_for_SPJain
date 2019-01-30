import pandas as pd
import nltk


df = pd.read_excel('Question_Answers.xlsx')
df.Questions = df.Questions.str.lower()


def pos_tagging(sentence):
    postag = nltk.pos_tag(nltk.word_tokenize(sentence))
    word = []
    for group in postag:
        if group[1][0]=='N' or group[1][:2] == 'JJ':
            word.append(group[0])                        
    return " ".join(word)


df["Nouns"]=df.Questions.apply(pos_tagging)
df.to_excel('Data.xlsx')
