
# coding: utf-8

# In[2]:

import pandas as pd


# In[3]:

df = pd.read_excel('Question_Answers.xlsx')


# In[3]:

df.Questions = df.Questions.str.lower()


# In[4]:

import nltk
temp=[]
for i in list(df['Questions']):
    temp.append(nltk.word_tokenize(i))

postag = []
for i in temp:
    postag.append(nltk.pos_tag(i))
    

Nouns = []

for row in postag:
    N = []
    for num in range(len(row)):
        word = row[num]
        if word[1][0] == 'N' or word[1][:2] == 'JJ' :
            N.append(word[0])
    Nouns.append(N)

Nouns=pd.DataFrame(Nouns)
Nouns['Nouns'] = Nouns[0].map(str) + ' ' + Nouns[1].map(str) + ' ' + Nouns[2].map(str) + ' ' + Nouns[3].map(str) + ' ' + Nouns[4].map(str) + ' ' + Nouns[5].map(str)
Nouns.drop(Nouns[[0,1,2,3,4,5]], axis=1, inplace=True)
for i in range(len(Nouns)):
    Nouns.Nouns.loc[i] = Nouns.Nouns.loc[i].replace(' None', '')


# In[5]:

df['Nouns'] = Nouns


# In[6]:

df.to_excel('Data.xlsx')


# In[ ]:




# In[ ]:



