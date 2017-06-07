
# coding: utf-8

# In[29]:

import pandas as pd


# In[30]:

df = pd.read_excel('Data.xlsx')


# In[31]:

import warnings
warnings.filterwarnings('ignore')


# ### Vectorising Questions 

# In[32]:

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

# In[33]:

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


# In[34]:

test2C = test2.copy()


# In[35]:

test2C.columns = range(test.shape[1], test.shape[1]+ test2.shape[1])


# In[36]:

test2.index = range(len(test2))
test.index = range(len(test))
test2C.index = range(len(test2C))


# In[37]:

test3 = pd.concat([test.reset_index(drop=True), test2C], axis=1)

#test has vectorised questions in dataframe. test2 does the same for nouns. test3 dataframe concatenates test and test2


# In[38]:

test['Question'] = list(df.Questions)
test['Answer'] = list(df.Answers)


# In[39]:

test2['Question'] = list(df.Questions)
test2['Answer'] = list(df.Answers)


# In[40]:

test3['Question'] = list(df.Questions)
test3['Answer'] = list(df.Answers)


# In[ ]:




# In[ ]:




# <br><br>

# ### Applying Random Forest 

# In[41]:

TestDF = pd.DataFrame()
TrainDF = pd.DataFrame()
def RFModel(df, test_size=0.1):
    global TestDF
    global TrainDF
    global model
    from sklearn.metrics import accuracy_score
    df = df[df.Answer.isnull() == False]
    df.index = range(len(df))
    from sklearn.cross_validation import train_test_split
    X = df[list(range(df.shape[1]-2))]
    y = df['Answer']
    y = y.astype('category')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    #Import Library
    from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression problem
    #Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
    # Create Random Forest object
    model= RandomForestClassifier(max_features= 'auto' ,n_estimators= 1000)
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
        print('Test accuracy is {}'.format(testAcc))
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
    print('Training accuracy is {}'.format(trainAcc))


# In[42]:

RFModel(test2)


# In[43]:

TestDF.to_excel('new.xlsx')


# In[44]:

TrainDF.to_excel('trainRF.xlsx')


# In[45]:

RFModel(test2, test_size=0)


# ### Creating rules, fallback responses etc. 

# In[46]:

courseNames = {
    'bdva' : ['big data & visual analytics', 'big data & analytics', 'big data and visual analytics', 'big data and analytics'  'big data analytics', 'data analytics','business analytics', 'big data', 'data science', 'analytics', 'data mining', 'data', 'visualisation','machine learning', 'artificial intelligence', 'bdap', 'bdva', 'bdvap'],
    'dmm' : ['digital marketing & metrics', 'digital marketing and metrics', 'digital marketing', 'digital metrics', 'dmm'],
    'rm' : ['retail management', 'retail', ' rm'],
    'emba': ['executive master of business administration', 'executive masters of business administration', 'executive master in business administration', 'executive masters in business administration', 'executive mba', 'executive', 'emba'],
    'gmba': ['global masters of business administration', 'global master of business administration', 'global masters in business administration', 'global master in business administration', 'global mba', 'globalmba', 'gmba'],
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
    'bdva' : 'You can have a word with Mr. Rakesh Shetty, M: +91- 82912 87131 | Email id rakesh.shetty@spjain.org. He handles all the application for the Big Data & Visual Analytics Course.',
    'dmm' : 'I would like to inform you that we have our Admissions Manager Mr. Ashutosh Kulkarni, based out of Mumbai.  He handles all applications for Digital Marketing & Metrics . You can get in touch with him directly. His contact details are tel no +91 9702896668, email id ashutosh.kulkarni@spjain.org . He would be your point of contact as far as DMM @ S P Jain is concerned.',
    'rm' : 'You can have a word with Mr. Ashutosh Kulkarni, M: +91 9702896668, email id ashutosh.kulkarni@spjain.org. He handles all applications for Retail Management.',
    'emba': 'I would like to inform you that we have our Assistant Manager- Professional Programs for EMBA Ms. Nikita Marwaha , based out of our Mumbai Campus. Ms. Nikita handles all applications for the EMBA program. You can get in touch with her directly. Her contact details are tel no 8879866774 email id nikita.marwaha@spjain.org. She would be your point of contact as far as EMBA @ S P Jain is concerned.',
    'gmba': 'I would like to inform you that we have our Admissions Manager Ms. Mona Dabas, based out of New Delhi. Ms. Mona handles all applications for MGB & GMBA coming from your region. You can get in touch with her directly. Her contact details are tel no +91 9818206168. Email id mona.dabas@spjain.org. She would be your point of contact as far as PG @ S P Jain is concerned',
    'mgb' : 'I would like to inform you that we have our Admissions Manager Mr. Noel Thomas, based out of Mumbai. Mr. Noel handles all applications for MGB & GMBA coming from your region. You can get in touch with him directly. His contact details are tel no +91 9322240630, email id noel.thomas@spjain.org. He would be your point of contact as far as PG @ S P Jain is concerned.',
    'mba' : 'I would like to inform you that we have our Admissions Manager Mr. Noel Thomas, based out of Mumbai. Mr. Noel handles all applications for MGB & GMBA coming from West India You can get in touch with him directly. His contact details are tel no +91 9322240630, email id noel.thomas@spjain.org. He would be your point of contact as far as PG @ S P Jain is concerned.',
    'dba' : 'I would like to inform you that we have our Director- Professional PG Programs Dr. Raja Roy Choudhury, based out of Mumbai.  He handles all applications for DMM/DBA. You can get in touch with him directly. His contact details are tel no 8879489681/8655502525, email id raja.royc@spjain.org  . He would be your point of contact as far as DBA/DMM @ S P Jain is concerned.',
    'GFMB' : 'You can have a word with Ms. Tejal Dhulla, M: 9987081818|Email id tejal.dhulla@spjain.org. She handles all the application for the GFMB. She would be your point of contact as far as GFMB @ S P Jain is concerned.',
    'bba' : 'I would like to inform you that we have our Admissions Manager Mr. Atharv Kale, based in Mumbai.  Atharv handles all applications for BBA, BBC & BEC coming from your region. You can get in touch with him directly. His contact details are Tel no +91 9920211171, email id: atharv.kale@spjain.org. He would be your point of contact as far as BBA, BBC & BEC @ S P Jain is concerned.',
    'bec' : 'I would like to inform you that we have our Admissions Manager Mr. Atharv Kale, based in Mumbai.  Atharv handles all applications for BBA, BBC & BEC coming from your region. You can get in touch with him directly. His contact details are Tel no +91 9920211171, email id: atharv.kale@spjain.org. He would be your point of contact as far as BBA, BBC & BEC @ S P Jain is concerned.',
    'bbc' : 'You can have a word with Mr. Mahesh Sharma, M: +91-7875874565  | Email id mahesh.sharma@spjain.org. He handles all the application for the Bachelor Of Business Communication.',
    'mgluxm': 'I would like to inform you that we have our Admissions Manager Ms. Neha Sikri , based out of Mumbai.  She handles all applications for MGLuxM. You can get in touch with her directly. Her contact details are tel no +91 9867633315, email id Neha.sikri@spjain.org. She would be your point of contact as far as MGLuxM @ S P Jain is concerned.',
    'fintech': 'I would like to inform you that we have our Admissions Manager Mr.Nikhil Gore , based out of Mumbai.  He handles all applications for Fintech  . You can get in touch with him directly. His contact details are tel no +91 9445431213, email id nikhil.gore@spjain.org. He would be your point of contact as far as DMM @ S P Jain is concerned.'
    
}

greetings = [' hi', ' hello', ' how are you', ' are you there', ' there?', ' hey', ' good morning', ' good afternoon', ' good evening', ' yo ']

thanks = ['thanks', 'thank', 'thx', 'thnx', 'thnks', 'thnk', 'bye', 'okay', 'ok']


# In[ ]:




# In[47]:

def checkQ(j):
    if '?' in j or j[0:5].lower() =='what ' or j[0:2].lower()=='do' or j[0:3].lower()=='can'     or j[0:2].lower() == 'is' or j[0:5].lower()=='would' or j[:4].lower() == 'how ':
        return True
    else:
        return False

def removePunct(S):
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    return ' '.join(tokenizer.tokenize(S))


# <br><br>

# Defining the chatbot function. Integrates random forest predictions, rules, fallback answers etc. It takes user input, responds to it, and asks for further input from user. Input 'quit' to stop the function.

# In[48]:

def chatbot():
    c = ''
    prev = ''
    on = False
    while on == False:
        new = False
        q = input().lower()
        if checkQ(q) == False:
            for i in thanks:
                if i in removePunct(q).split():
                    new = True
                    prev = ''
                    print('SP Jain Assistant: ' + "It's my pleasure to have helped you. All the best!" + '\n')
                    break
                    
        q = removePunct(q) + prev
        query = c + ' ' + q
        

        if new:
            continue
            
        if q == 'quit':
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
                            find = True
                            new = True
                            prev = ''
                            query = q.replace(name, ' ' + str(course) + ' ')
                            #Q = [tf(word, query) for word in vocabulary]
                            N = [tf(word, query) for word in vocabulary2]
                            prob = max(max(model.predict_proba(N)))
                            if prob>0.35:
                                print('SP Jain Assistant: ' + model.predict(N)[0] + '\n')
                            elif prob >= 0.22 and model.predict(N)[0] != defaultResp[course] + ' ':
                                print('SP Jain Assistant: ' + model.predict(N)[0] + '\n' + defaultResp[course] + '\n')
                            else:
                                print('SP Jain Assistant: ' + defaultResp[course] + '\n')
                            c = ' ' + course
                            break
                            
                        elif counter == 97:
                            for course in sorted(courseNames):
                                if fin:
                                    break
                                else:
                                    for name in courseNames[course]:
                                        if name in query:
                                            prev = ''
                                            fin = True
                                            new = True
                                            query = query.replace(name, ' ' + course + ' ')
                                            #Q = [tf(word, query) for word in vocabulary]
                                            N = [tf(word, query) for word in vocabulary2]
                                            prob = max(max(model.predict_proba(N)))
                                            if prob>0.35:
                                                print('SP Jain Assistant: ' + model.predict(N)[0] + '\n')
                                            elif prob >= 0.22 and model.predict(N)[0] != defaultResp[course] + ' ':
                                                print('SP Jain Assistant: ' + model.predict(N)[0] + '\n' + defaultResp[course] + '\n')
                                            else:
                                                print('SP Jain Assistant: ' + defaultResp[course] + '\n')
                                            break
                        

            if find == False and fin == False:   
                if probability > 0.4:
                    new = True
                    prev = ''
                    print('SP Jain Assistant: ' + model.predict(N)[0] + '\n')


                if new == False:    
                    for word in greetings:
                        if word in query:
                            new = True
                            prev = ''
                            print('SP Jain Assistant: ' + 'Hello! How may I help you?' + '\n')
                            break

                        elif word == greetings[-1]:
                            new = True
                            print('SP Jain Assistant: ' + 'May I know which program you are looking for?' + '\n')
                            prev = q




# <br><br><br><br><br><br><br>

# In[49]:

import warnings
warnings.filterwarnings('ignore')


# In[50]:

chatbot()

