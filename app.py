
# coding: utf-8

# In[66]:

import pandas as pd


# In[67]:

df = pd.read_excel('Data.xlsx')


# In[68]:

import warnings
warnings.filterwarnings('ignore')


# ### Vectorising Questions 

# In[69]:

import string #allows for format()
import pandas

mydoclist = list(df.Questions)

from collections import Counter
for doc in mydoclist:
    tf = Counter()
    for word in doc.split():
        tf[word] +=1
    #print (tf.items())    
def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

def tf(term, document):
    return freq(term, document)

def freq(term, document):
    return document.split().count(term)

vocabulary = build_lexicon(mydoclist)

doc_term_matrix = []
#print ('Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']')
for doc in mydoclist:
    #print ('The doc is "' + doc + '"')
    tf_vector = [tf(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    #print ('The tf vector for Document %d is [%s]' % ((mydoclist.index(doc)+1), tf_vector_string))
    doc_term_matrix.append(tf_vector)

    # here's a test: why did I wrap mydoclist.index(doc)+1 in parens?  it returns an int...
    # try it!  type(mydoclist.index(doc) + 1)

#print ('All combined, here is our master document term matrix: ')
#print (doc_term_matrix)

for doc in mydoclist:
    tf_vector = [tf(word, doc) for word in vocabulary]
    #print(tf_vector)

t=pandas.DataFrame(columns=[i for i in range(0,1348) ])
cntr = 0
for doc in mydoclist:
    #print ('The doc is "' + doc + '"')
    tf_vector = [tf(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    #print (tf_vector_string)
    c=[i for i in range(1348)]
    #print(tf_vector)
    if cntr == 0:
        test = pandas.DataFrame([tf_vector])
        #print(cntr)
        #print(test)
    else:
        test = test.append([tf_vector])
#         print(cntr)
#         print(test)
    cntr += 1
    #print(cntr)





# ### Vectorising Key Words in Question

# In[70]:

import string #allows for format()
import pandas

mydoclist = list(df.Nouns)

from collections import Counter
for doc in mydoclist:
    tf = Counter()
    for word in doc.split():
        tf[word] +=1
    #print (tf.items())    
def build_lexicon(corpus):
    lexicon = set()
    for doc in corpus:
        lexicon.update([word for word in doc.split()])
    return lexicon

def tf(term, document):
    return freq(term, document)

def freq(term, document):
    return document.split().count(term)

vocabulary2= build_lexicon(mydoclist)

doc_term_matrix = []
#print ('Our vocabulary vector is [' + ', '.join(list(vocabulary)) + ']')
for doc in mydoclist:
    #print ('The doc is "' + doc + '"')
    tf_vector = [tf(word, doc) for word in vocabulary2]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    #print ('The tf vector for Document %d is [%s]' % ((mydoclist.index(doc)+1), tf_vector_string))
    doc_term_matrix.append(tf_vector)

    # here's a test: why did I wrap mydoclist.index(doc)+1 in parens?  it returns an int...
    # try it!  type(mydoclist.index(doc) + 1)

#print ('All combined, here is our master document term matrix: ')
#print (doc_term_matrix)

for doc in mydoclist:
    tf_vector = [tf(word, doc) for word in vocabulary2]
    #print(tf_vector)

t=pandas.DataFrame(columns=[i for i in range(0,1348) ])
cntr = 0
for doc in mydoclist:
    #print ('The doc is "' + doc + '"')
    tf_vector = [tf(word, doc) for word in vocabulary2]
    f_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    #print (tf_vector_string)
    c=[i for i in range(1348)]
    #print(tf_vector)
    if cntr == 0:
        test2= pandas.DataFrame([tf_vector])
        #print(cntr)
        #print(test)
    else:
        test2= test2.append([tf_vector])
        #print(cntr)
        #print(test)
    cntr += 1
    #print(cntr)


# In[71]:

test2C = test2.copy()


# In[72]:

test2C.columns = range(test.shape[1], test.shape[1]+ test2.shape[1])


# In[73]:

test2.index = range(len(test2))
test.index = range(len(test))
test2C.index = range(len(test2C))


# In[74]:

test3 = pd.concat([test.reset_index(drop=True), test2C], axis=1)

#test has vectorised questions in dataframe. test2 does the same for nouns. test3 dataframe concatenates test and test2


# In[75]:

test['Question'] = list(df.Questions)
test['Answer'] = list(df.Answers)


# In[76]:

test2['Question'] = list(df.Questions)
test2['Answer'] = list(df.Answers)


# In[77]:

test3['Question'] = list(df.Questions)
test3['Answer'] = list(df.Answers)


# In[ ]:




# In[ ]:




# <br><br>

# ### Applying Random Forest 

# In[78]:

TestDF = pd.DataFrame()
TrainDF = pd.DataFrame()
def RFModel(df, test_size=0.1):
    global TestDF
    global TrainDF
    global model
    from sklearn.metrics import accuracy_score
    df = df[df.Answer.isnull() == False]
    df.index = range(len(df))
    from sklearn.model_selection import train_test_split
    X = df[list(range(df.shape[1]-2))]
    y = df['Answer']
    y = y.astype('category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    #Import Library
    from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression problem
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Random Forest object
    model= RandomForestClassifier(max_features= 'auto' ,n_estimators= 200)
    # Train the model using the training sets and check score
    #model.fit(X, y)
    #from sklearn import tree
    #model = tree.DecisionTreeClassifier(criterion='gini')
    model.fit(X_train, y_train)
    model.score(X_train, y_train)
    #Predict Output
    if test_size > 0.01:
        testPredicted= model.predict(X_test)
        test = pd.DataFrame(df.loc[list(X_test.index)])
        TestDF = test[['Question']]
        TestDF['Prediction'] = testPredicted
        ps = model.predict_proba(X_test)
        prob = []
        for i in ps:
            prob.append(max(i))
        TestDF['Prediction Probability'] = prob
        testAcc = accuracy_score(y_test, testPredicted)
        #print('Test accuracy is {}'.format(testAcc))
    trainPredicted = model.predict(X_train)
    train = pd.DataFrame(df.loc[list(X_train.index)])
    TrainDF = train[['Question']]
    TrainDF['Prediction'] = trainPredicted
    tprob = []
    ps = model.predict_proba(X_train)
    for i in ps:
        tprob.append(max(i))
    TrainDF['Prediction Probability'] = tprob
    trainAcc = accuracy_score(y_train, trainPredicted)
    #print('Training accuracy is {}'.format(trainAcc))


# In[79]:

#RFModel(test2)


# In[80]:

TestDF.to_excel('new.xlsx')


# In[81]:

TrainDF.to_excel('trainRF.xlsx')


# In[82]:

import warnings
warnings.filterwarnings('ignore')


# In[83]:

RFModel(test2, test_size=0)


# ### Creating rules, fallback responses etc. 

# In[1]:

courseNames = {
    'bdva' : ['big data & visual analytics', 'big data & analytics', 'big data and visual analytics', 'big data and analytics'  'big data analytics', 'data analytics','business analytics', 'big data', 'data science', 'analytics', 'data mining', 'data', 'visualisation','machine learning', 'artificial intelligence', 'bdap', 'bdva', 'bdvap'],
    'dmm' : ['digital marketing & metrics', 'digital marketing and metrics', 'digital marketing', 'digital metrics', 'dmm'],
    'rm' : ['retail management', 'retail', ' rm'],
    'emba': ['executive master of business administration', 'executive masters of business administration', 'executive master in business administration', 'executive masters in business administration', 'executive mba', 'executive', 'emba'],
    'gmba': ['global masters of business administration', 'global master of business administration', 'global masters in business administration', 'global master in business administration', 'global mba', 'globalmba', 'gmba', 'mba'],
    'mgb' : ['masters in global business', 'masters of global business', 'master of global business', 'masters in global business', 'master in global business', 'mgb'],
    'mba' : ['masters of business administration',' mba'],
    'dba' : ['doctor of business administration', 'doctorate of business administration', 'business administration doctorate', 'doctor in business administration', 'business doctorate', 'doctorate', 'doctor', 'dba'],
    'gfmb' : ['global family managed business', 'family managed business', 'family business', 'global fmb', 'gfmb', 'fmb'],
    'bba' : ['bachelor of business administration', 'bachelor in business administration', 'bachelors in business administration', 'bachelors of business administration', 'business management', 'bba'],
    'bec' : ['bachelor of economics', 'bachelor in economics', 'bachelors in economics', 'bachelors of economics', 'economics', 'bachelor of eco', 'bec'],
    'bbc' : ['bachelor of business communication', 'bachelors of business communication', 'bachelor in business communication', 'bachelors in business communication', 'business communication', 'bbc'],
    'mgluxm': ['masters in global luxury goods and services management', 'masters in global luxury management goods and services', 'masters in global luxury management', 'masters in global luxury goods and services', 'masters in global luxury management goods', 'global luxury management', 'luxury management of goods and services', 'luxury management of goods & services' 'luxury management', 'luxury goods and services management', 'luxury goods and services', 'luxury', 'mgluxm'],
    'fintech': ['financial technology', 'finance technology', 'professional finance course', 'finance course', 'fintech']              
}

defaultResp = {
    'bdva' : 'You can have a word with Mr. Rakesh Shetty, M: +91- 82912 87131 | Email id rakesh.shetty@spjain.org. He handles all the application for the Big Data and Visual Analytics Course.',
    'dmm' : 'I would like to inform you that we have our Admissions Manager Mr. Ashutosh Kulkarni, based out of Mumbai.  He handles all applications for Digital Marketing and Metrics . You can get in touch with him directly. His contact details are tel no +91 9702896668, email id ashutosh.kulkarni@spjain.org . He would be your point of contact as far as DMM @ S P Jain is concerned.',
    'rm' : 'You can have a word with Mr. Ashutosh Kulkarni, M: +91 9702896668, email id ashutosh.kulkarni@spjain.org. He handles all applications for Retail Management.',
    'emba': 'I would like to inform you that we have our Assistant Manager- Professional Programs for EMBA Ms. Nikita Marwaha , based out of our Mumbai Campus. Ms. Nikita handles all applications for the EMBA program. You can get in touch with her directly. Her contact details are tel no 8879866774 email id nikita.marwaha@spjain.org. She would be your point of contact as far as EMBA @ S P Jain is concerned.',
    'gmba': 'I would like to inform you that we have our Admissions Manager Ms. Mona Dabas, based out of New Delhi. Ms. Mona handles all applications for MGB and GMBA coming from your region. You can get in touch with her directly. Her contact details are tel no +91 9818206168. Email id mona.dabas@spjain.org. She would be your point of contact as far as PG @ S P Jain is concerned',
    'mgb' : 'I would like to inform you that we have our Admissions Manager Mr. Noel Thomas, based out of Mumbai. Mr. Noel handles all applications for MGB and GMBA coming from your region. You can get in touch with him directly. His contact details are tel no +91 9322240630, email id noel.thomas@spjain.org. He would be your point of contact as far as PG @ S P Jain is concerned.',
    'mba' : 'I would like to inform you that we have our Admissions Manager Mr. Noel Thomas, based out of Mumbai. Mr. Noel handles all applications for MGB and GMBA coming from West India You can get in touch with him directly. His contact details are tel no +91 9322240630, email id noel.thomas@spjain.org. He would be your point of contact as far as PG @ S P Jain is concerned.',
    'dba' : 'I would like to inform you that we have our Director- Professional PG Programs Dr. Raja Roy Choudhury, based out of Mumbai.  He handles all applications for DMM/DBA. You can get in touch with him directly. His contact details are tel no 8879489681/8655502525, email id raja.royc@spjain.org  . He would be your point of contact as far as DBA/DMM @ S P Jain is concerned.',
    'GFMB' : 'You can have a word with Ms. Tejal Dhulla, M: 9987081818|Email id tejal.dhulla@spjain.org. She handles all the application for the GFMB. She would be your point of contact as far as GFMB @ S P Jain is concerned.',
    'bba' : 'I would like to inform you that we have our Admissions Manager Mr. Atharv Kale, based in Mumbai.  Atharv handles all applications for BBA, BBC and BEC coming from your region. You can get in touch with him directly. His contact details are Tel no +91 9920211171, email id: atharv.kale@spjain.org. He would be your point of contact as far as BBA, BBC and BEC @ S P Jain is concerned.',
    'bec' : 'I would like to inform you that we have our Admissions Manager Mr. Atharv Kale, based in Mumbai.  Atharv handles all applications for BBA, BBC and BEC coming from your region. You can get in touch with him directly. His contact details are Tel no +91 9920211171, email id: atharv.kale@spjain.org. He would be your point of contact as far as BBA, BBC and BEC @ S P Jain is concerned.',
    'bbc' : 'You can have a word with Mr. Mahesh Sharma, M: +91-7875874565  | Email id mahesh.sharma@spjain.org. He handles all the application for the Bachelor Of Business Communication.',
    'mgluxm': 'I would like to inform you that we have our Admissions Manager Ms. Neha Sikri , based out of Mumbai.  She handles all applications for MGLuxM. You can get in touch with her directly. Her contact details are tel no +91 9867633315, email id Neha.sikri@spjain.org. She would be your point of contact as far as MGLuxM @ S P Jain is concerned.',
    'fintech': 'I would like to inform you that we have our Admissions Manager Mr.Nikhil Gore , based out of Mumbai.  He handles all applications for Fintech  . You can get in touch with him directly. His contact details are tel no +91 9445431213, email id nikhil.gore@spjain.org. He would be your point of contact as far as DMM @ S P Jain is concerned.'
    
}

courseList = ['- ' + i[0].title().replace('&', 'and') for i in courseNames.values()]

greetings = [' hi ', ' hello ', ' how are you', ' are you there', ' there?', ' hey ', ' good morning', ' good afternoon', ' good evening', ' yo ']

thanks = ['thanks', 'thank', 'thx', 'thnx', 'thnks', 'thnk','thanka', 'bye', 'okay', 'ok', 'cool', 'sure']


# In[2]:

def checkQ(j):
    if '?' in j or 'what ' in j or 'do' in j or 'can' in j     or j[0:2].lower() == 'is' or j[0:5].lower()=='would' or 'how ' in j:
        return True
    else:
        return False

def removePunct(S):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    return ' '.join(tokenizer.tokenize(S))


# <br><br>

# Defining the chatbot function. Integrates random forest predictions, rules, fallback answers etc. It takes user input, responds to it, and asks for further input from user. Input 'quit' to stop the function.

# In[3]:

# def chatbot():
#     c = ''
#     prev = ''
#     on = False
#     while on == False:
#         new = False
#         q = input().lower()
#         if checkQ(q) == False:
#             for i in thanks:
#                 if i in removePunct(q).split():
#                     new = True
#                     prev = ''
#                     print('SP Jain Assistant: ' + "It's my pleasure to have helped you. All the best!" + '\n')
#                     break
                    
#         q = removePunct(q) + ' ' + prev
#         query = c + ' ' + q + ' ' + prev
        

#         if new:
#             continue
            
#         if q == 'quit ' + prev:
#             on = True
            
#         else:
#             N = [tf(word, query) for word in vocabulary2]
#             probability = max(max(model.predict_proba(N)))
#             find = False
#             fin = False
#             counter = 0
#             for course in sorted(courseNames):
#                 if find:
#                     break
#                 else:    
#                     for name in courseNames[course]:
#                         counter +=1
#                         if name in q:
#                             find = True
#                             new = True
#                             prev = ''
#                             query = q.replace(name, ' ' + str(course) + ' ')
#                             #Q = [tf(word, query) for word in vocabulary]
#                             N = [tf(word, query) for word in vocabulary2]
#                             prob = max(max(model.predict_proba(N)))
#                             if prob>0.35:
#                                 print('SP Jain Assistant: ' + model.predict(N)[0] + '\n')
#                             elif prob >= 0.22 and model.predict(N)[0] != defaultResp[course] + ' ':
#                                 print('SP Jain Assistant: ' + model.predict(N)[0] + '\n' + defaultResp[course] + '\n')
#                             else:
#                                 print('SP Jain Assistant: ' + defaultResp[course] + '\n')
#                             c = ' ' + course
#                             break
                            
#                         elif counter == 97:
#                             for course in sorted(courseNames):
#                                 if fin:
#                                     break
#                                 else:
#                                     for name in courseNames[course]:
#                                         if name in query:
#                                             prev = ''
#                                             fin = True
#                                             new = True
#                                             query = query.replace(name, ' ' + course + ' ')
#                                             #Q = [tf(word, query) for word in vocabulary]
#                                             N = [tf(word, query) for word in vocabulary2]
#                                             prob = max(max(model.predict_proba(N)))
#                                             if prob>0.35:
#                                                 print('SP Jain Assistant: ' + model.predict(N)[0] + '\n')
#                                             elif prob >= 0.22 and model.predict(N)[0] != defaultResp[course] + ' ':
#                                                 print('SP Jain Assistant: ' + model.predict(N)[0] + '\n' + defaultResp[course] + '\n')
#                                             else:
#                                                 print('SP Jain Assistant: ' + defaultResp[course] + '\n')
#                                             break
                        

#             if find == False and fin == False:   
#                 if probability > 0.4:
#                     new = True
#                     prev = ''
#                     print('SP Jain Assistant: ' + model.predict(N)[0] + '\n')


#                 if new == False:    
#                     for word in greetings:
#                         if word in query:
#                             new = True
#                             prev = ''
#                             print('SP Jain Assistant: ' + 'Hello! How may I help you?' + '\n')
#                             break

#                         elif word == greetings[-1]:
#                             new = True
#                             print('SP Jain Assistant: ' + 'May I know which program you are looking for?' + '\n')
#                             prev = q




# <br><br><br><br><br><br><br>

# In[4]:

import warnings
warnings.filterwarnings('ignore')


# In[5]:

# chatbot()


# In[6]:

chatTracker = {}

import random


# In[10]:

#rep = False
#Help = False
#repeat = False

def spjBot(q, chat, c, prev, rep, Help, repeat, progQ):
    #global rep
    #global Help
    #global repeat
    on = False
    return "testing"
    while on == False:
        new = False
        q = ' ' + q.lower() + ' '
        if checkQ(q) == False:
            for i in thanks:
                if i in removePunct(q).split():
                    chatTracker[chat]['progQ'] = False
                    new = True
                    prev = ''
                    chatTracker[chat]['prev'] = ''
                    rep = False
                    chatTracker[chat]['rep'] = False
                    if Help == True and thanks.index(i) < 7:
                        return "It's my pleasure to have helped you. All the best!" + '\n'
                    else:
                        return 'Good Luck!'
                    
            for i in greetings:
                if i in ' ' + removePunct(q) + ' ':
                    chatTracker[chat]['progQ'] = False
                    new = True
                    prev = ''
                    chatTracker[chat]['prev'] = ''
                    rep = False
                    chatTracker[chat]['rep'] = False
                    return 'Hi. Let me know if you need any help.'
                    
        q = removePunct(q) + ' ' + chatTracker[chat]['prev']
        query = chatTracker[chat]['c'] + ' ' + q + ' ' + chatTracker[chat]['prev']
        

        if new:
            continue
            
        if q == 'quit ' + prev:
            on = True
            
        else:
            N = [tf(word, query) for word in vocabulary2]
            probability = max(max(model.predict_proba(N)))
            find = False
            fin = False
            counter = 0
            for course in sorted(courseNames):
                if find:
                    break
                else:    
                    for name in courseNames[course]:
                        counter +=1
                        if name in q:
                            chatTracker[chat]['progQ'] = False
                            find = True
                            new = True
                            Help = True
                            chatTracker[chat]['Help'] = True
                            prev = ''
                            chatTracker[chat]['prev'] = ''
                            chatTracker[chat]['c'] = ' ' + course
                            query = q.replace(name, ' ' + str(course) + ' ')
                            #Q = [tf(word, query) for word in vocabulary]
                            N = [tf(word, query) for word in vocabulary2]
                            prob = max(max(model.predict_proba(N)))
                            if prob>0.35:
                                rep = False
                                chatTracker[chat]['rep'] = False
                                return model.predict(N)[0] + '\n'
                            elif (len(name.split()) == len(q.split()) or sum(N) < 2) and probability < 0.3:
                                chatTracker[chat]['progQ'] = False
                                return "What do you want to know about the " + courseNames[course][0].replace('&', 'and') + " program? \nYou can ask about the fees, duration, scholarships and so on"
                            elif prob >= 0.22 and model.predict(N)[0] != defaultResp[course] + ' ':
                                if rep == False:
                                    #rep = True
                                    #chatTracker[chat]['rep'] = True
                                    #repeat = True
                                    #chatTracker[chat]['repeat'] = True
                                    chatTracker[chat]['c'] = ''
                                    return model.predict(N)[0] + '\n' + defaultResp[course] + '\n'
                                else:
                                    rep = False
                                    chatTracker[chat]['rep'] = False
                                    Help = False
                                    chatTracker[chat]['Help'] = False
                                    chatTracker[chat]['c'] = ''
                                    return 'Our course representative may be able to help you on this.'
                            else:
                                if rep == False:
                                    #repeat = True
                                    #chatTracker[chat]['repeat'] = True
                                    #rep = True
                                    #chatTracker[chat]['rep'] = True
                                    chatTracker[chat]['c'] = ''
                                    return defaultResp[course] + '\n'
                                else:
                                    rep = False
                                    chatTracker[chat]['rep'] = False
                                    Help = False
                                    chatTracker[chat]['Help'] = False
                                    chatTracker[chat]['c'] = ''
                                    return 'Our course representative may be able to help you on this.'
                            #break
                            
                        elif counter == 97:
                            for course in sorted(courseNames):
                                if fin:
                                    break
                                else:
                                    for name in courseNames[course]:
                                        if name in query:
                                            chatTracker[chat]['progQ'] = False
                                            prev = ''
                                            chatTracker[chat]['prev'] = ''
                                            fin = True
                                            new = True
                                            Help = True
                                            chatTracker[chat]['Help'] = True
                                            query = query.replace(name, ' ' + course + ' ')
                                            #Q = [tf(word, query) for word in vocabulary]
                                            N = [tf(word, query) for word in vocabulary2]
                                            prob = max(max(model.predict_proba(N)))
                                            if prob>0.35:
                                                return model.predict(N)[0] + '\n'
                                            elif prob >= 0.22 and model.predict(N)[0] != defaultResp[course] + ' ':
                                                if repeat == False:
                                                    #repeat = True
                                                    #chatTracker[chat]['repeat'] = True
                                                    #rep = True
                                                    #chatTracker[chat]['rep'] = True
                                                    chatTracker[chat]['c'] = ''
                                                    return model.predict(N)[0] + '\n' + defaultResp[course] + '\n'
                                                else:
                                                    Help = False
                                                    chatTracker[chat]['Help'] = False
                                                    chatTracker[chat]['c'] = ''
                                                    chatTracker[chat]['repeat'] = False
                                                    return 'Our course representative may be able to help you on this.'
                                            else:
                                                if repeat == False:
                                                    #repeat = True
                                                    #chatTracker[chat]['repeat'] = True
                                                    #rep = True
                                                    #chatTracker[chat]['rep'] = True
                                                    chatTracker[chat]['c'] = ''
                                                    return defaultResp[course] + '\n'
                                                else:
                                                    Help = False
                                                    chatTracker[chat]['Help'] = False
                                                    chatTracker[chat]['c'] = ''
                                                    chatTracker[chat]['repeat'] = False
                                                    return 'Our course representative may be able to help you on this.'
                                                #break
                        

            if find == False and fin == False:   
                if probability > 0.4:
                    Help = True
                    chatTracker[chat]['Help'] = True
                    new = True
                    prev = ''
                    rep = False
                    chatTracker[chat]['rep'] = False
                    chatTracker[chat]['prev'] = ''
                    return model.predict(N)[0] + '\n'


                if new == False:
 
                    for word in greetings:
                        if word in query:
                            if checkQ(q) == False:
                                new = True
                                prev = ''
                                rep = False
                                chatTracker[chat]['rep'] = False
                                chatTracker[chat]['prev'] = ''
                                return 'Hello! How may I help you?' + '\n'
                            else:
                                new = True
                                chatTracker[chat]['prev'] = q
                                rep = False
                                chatTracker[chat]['rep'] = False
                                chatTracker[chat]['progQ'] = True
                                return 'May I know which program you are looking for?' + '\n\n' + "Some of the various courses we offer are:\n\n" + '\n'.join(random.sample(courseList, 3)) + "\n and so on..."
                                break

                        elif word == greetings[-1]:
                            if chatTracker[chat]['progQ'] == True:
                                chatTracker[chat]['progQ'] = False
                                Lc = [tf(word, "list of courses") for word in vocabulary2]
                                return "Well...\n " + model.predict(Lc)[0]
                            new = True
                            chatTracker[chat]['prev'] = q
                            rep = False
                            chatTracker[chat]['rep'] = False
                            chatTracker[chat]['progQ'] = True
                            return 'May I know which program you are looking for?' + '\n\n' + "Some of the various courses we offer are:\n\n" + '\n'.join(random.sample(courseList, 3)) + "\nand so on..."
                            




# In[ ]:






from flask import Flask, request
import json
import requests


app = Flask(__name__)

# This needs to be filled with the Page Access Token that will be provided
# by the Facebook App that will be created.
PAT = ''

@app.route('/', methods=['GET'])
def handle_verification():
    print("Handling Verification.")
    if request.args.get('hub.verify_token', '') == 'my_voice_is_my_password_verify_me':
        print("Verification successful!")
        return request.args.get('hub.challenge', '')
    else:
        print("Verification failed!")
        return 'Error, wrong validation token'

@app.route('/', methods=['POST'])
def handle_messages():
    print("Handling Messages")
    payload = request.get_data()
    print(payload)
    for sender, message in messaging_events(payload):
        print("Incoming from %s: %s" % (sender, message))
        send_message(PAT, sender, message)
    return "ok"

def messaging_events(payload):
    """Generate tuples of (sender_id, message_text) from the
    provided payload.
    """
    data = json.loads(payload)
    messaging_events = data["entry"][0]["messaging"]
    for event in messaging_events:
        if event['sender']['id'] not in chatTracker:
            ID = event['sender']['id']
            chatTracker[ID] = {}
            chatTracker[ID]['c'] = ''
            chatTracker[ID]['prev'] = ''

        if "message" in event and "text" in event["message"]:
            message = event["message"]["text"]
            ID = event["sender"]["id"]
            yield ID, spjBot(message, ID, chatTracker[ID]['c'], chatTracker[ID]['prev']).encode('unicode_escape')
        else:
            yield event["sender"]["id"], "I can't respond to this".encode('unicode_escape')


def send_message(token, recipient, text):
    """Send the message text to recipient with id recipient."""

    r = requests.post("https://graph.facebook.com/v2.6/me/messages",
    params={"access_token": token},
    data=json.dumps({
      "recipient": {"id": recipient},
      "message": {"text": text.decode('unicode_escape')}
    }),
    headers={'Content-type': 'application/json'})
    if r.status_code != requests.codes.ok:
        print(r.text)

if __name__ == '__main__':
    app.run()

