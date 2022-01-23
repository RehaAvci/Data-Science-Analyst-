#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn import linear_model


# In[2]:


data=pd.read_csv('auto-mpg.csv',sep=',',names=['0', '1','2','3','4','5','6','7'])
data.head(10)


# In[3]:


#For each independent variable, standardize with the sklearn StandardScaler: setting up scaler
scaler = StandardScaler()


# In[4]:


#For each independent variable, standardize with the sklearn StandardScaler: standardizing
data[["1",'2','3','4','5','6','7']]=scaler.fit_transform(data[["1",'2','3','4','5','6','7']])
data.head(10)


# In[5]:


#Split the data
#into 50% training and 50% test. Learn a multiple linear regression model. What is the
#mean squared error on the test set
Y = data['0'] #target variable is 0
X = data[["1",'2','3','4','5','6','7']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42069)
LinReg = LinearRegression().fit(X_train, Y_train)
prediction= LinReg.predict(X_test)
MSE = mean_squared_error(Y_test, prediction)
print(f'MSE = {MSE}')


# In[6]:


#Repeat part A, except scale the independent variables with sklearn MinMaxScaler. What
#is the mean squared error on the test set?
from sklearn.preprocessing import MinMaxScaler

scaler2 = MinMaxScaler()


# In[7]:


#For each independent variable, standardize with the sklearn StandardScaler: standardizing
data2=pd.read_csv('auto-mpg.csv',sep=',',names=['0', '1','2','3','4','5','6','7'])
data2[["1",'2','3','4','5','6','7']]=scaler2.fit_transform(data2[["1",'2','3','4','5','6','7']])
Y = data2['0'] #target variable is 0
X = data2[["1",'2','3','4','5','6','7']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42068)
LinReg2 = LinearRegression().fit(X_train, Y_train)
prediction= LinReg2.predict(X_test)
MSE = mean_squared_error(Y_test, prediction)
print(f'MSE = {MSE}')


# In[8]:


#Compare the coefficients from the class exercise and the models learned in part A and B.
#Which feature(s) are most predictive of MPG? Compare the weights of each feature
#before and after scaling.


# In[9]:


data3=pd.read_csv('auto-mpg.csv',sep=',',names=['0', '1','2','3','4','5','6','7'])
Y = data2['0'] #target variable is 0
X = data2[["1",'2','3','4','5','6','7']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42067)
LinReg3 = LinearRegression().fit(X_train, Y_train)
prediction= LinReg3.predict(X_test)


# In[10]:


LinRegs=pd.DataFrame(
    {'feature': ["1",'2','3','4','5','6','7'],
     'StandardScaler': LinReg.coef_,
     'MinMax': LinReg2.coef_,
     'No standardize': LinReg3.coef_
    })

LinRegs


# **I think fetures 6,4 are the most powerfull due to coef strength.

# In[11]:


#Using the MinMaxScaler on the independent variables, train a ridge regression model.
#How does the value for alpha change the model coefficients?


# In[12]:


data2=pd.read_csv('auto-mpg.csv',sep=',',names=['0', '1','2','3','4','5','6','7'])
data2[["1",'2','3','4','5','6','7']]=scaler2.fit_transform(data2[["1",'2','3','4','5','6','7']])
Y = data2['0'] #target variable is 0
X = data2[["1",'2','3','4','5','6','7']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42068)
LinReg2 = LinearRegression().fit(X_train, Y_train)
prediction= LinReg2.predict(X_test)
MSE = mean_squared_error(Y_test, prediction)

coef={}
for i in range(1,21):
    ridge = linear_model.Ridge(alpha=i*0.1)
    ridge.fit(X_train, Y_train)
    coef[i*0.1]=ridge.coef_

coef = pd.DataFrame(coef)
coef_T = coef.T

sns.lineplot(data=coef_T)


# In[13]:


coef={}
for i in range(1,21):
    lasso = linear_model.Lasso(alpha=i*0.1)
    lasso.fit(X_train, Y_train)
    coef[i*0.1]=lasso.coef_

coef = pd.DataFrame(coef)
coef_T = coef.T
sns.lineplot(data=coef_T)


# In[14]:


#a. Identify 10 non-word tokens in the passage.
#Thanks John!<br /><br /><font size="3"> &quot;Illustrations and demos will be provided for
#students to work through on their own&quot;</font>. Dowe need that to finish project? If yes, where to find
#the illustration and demos? Thanks for your help.\<img title="smile" alt="smile"
#src="\url{http://lms.statistics.com/pix/smartpix.php/statistics_com_1/s/smiley.gif}" \><br /> <br />
# <br />
# &quot
# </font>
# <font size="3">
# title="smile"
# alt="smile"
# url
# http
# gif
# php


# In[15]:


#b. Suppose that this passage constitutes a document to be classified, but you are not
#certain of the business goal of the classification task. Identify material (at least 20% of the
#terms) that, in your judgment, could be discarded fairly safely without knowing that goal.

#I think the words above in html format can be discarded. Words such as 
# <br />, <img title="smile" alt="smile" src="\url{http://lms.statistics.com/pix/smartpix.php/statistics_com_1/s/smiley.gif}" \><br /> <br />
# <font size="3">


# In[16]:


#c. Suppose that the classification task is to predict whether this post requires the attention
#of the instructor, or whether a teaching assistant might suffice. Identify the 20% of the
#terms that you think might be most helpful in that task.

# " thanks john illustrations demos provided students work need finish project yes find illustration demos thanks help "


# In[17]:


#d. What aspect of the passage is most problematic from the standpoint of simply using a
#bag-of-words approach, as opposed to an approach in which meaning is extracted?

#the html format that the paragraph is in makes bag-of-words approach somewhat difficult


# In[18]:


#Examine the raw text. What preprocessing techniques would be applicable to this data?
#(Examples of preprocessing: stemming, stopword removal, remove non-alphanumeric
#characters, entity recognition of URLs). Why will these preprocessing techniques be
#applicable?

data=pd.read_csv('SPAM text message.csv')
data.head(10)


# In[19]:


# In terms of which preprocessing technieues could be apllicable, 
# the techniques provided in the prompt are all applicable 
# in addition to Remove Extra Spaces, Remove punctuations, and Remove words and digits containing digits

# these preprocessing techniques are applicable because not all text related data is formatted correctly. RAW data 
# such as given dataset usually irregularities and undesired formatting requring preprocessing techniques to prepare
# the data to be worked with.


# In[20]:


# A preprocessing script is provided on the course homework webpage. Uncomment the
# preprocessing techniques you selected Part A. Run this script to 
# generate the termdocument matrix. Examine the term-document matrix.


# In[21]:


import pandas as pd
import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(891)
from nltk.corpus import stopwords
import re
import scipy.sparse

data = pd.read_csv('SPAM text message.csv', error_bad_lines=False)
data['spam'] = data.Category.map({'ham':0, 'spam':1})

stop = set(stopwords.words('english'))
#Create a stemmer
stemmer = SnowballStemmer(language = 'english')
#Create a lemmatizer
lemma = WordNetLemmatizer()

#Stem and lemmatize a term
def lemmatize_stemming(term):
    term = lemma.lemmatize(term, pos='v') # Lemmatize
    term = stemmer.stem(term) #Stem
    return term
    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text): #Deaccents and splits terms
        token = re.sub("\W","",token) #Remove special characters, punctuation, spaces
        
        token = token.lower() #lowercase string
        
        if token in stop: # Stopword removal: Remove "token not in stop" to keep stopwords
            token = ""
            
        if token.startswith("http"): #entity recognition of URLs.
            token = "URL_"
            
        if len(token) > 3:  
            result.append(lemmatize_stemming(token))
    return result

processed_docs = data['Message'].map(preprocess)

# Create a dictionary – word and its frequency in all documents
dictionary = gensim.corpora.Dictionary(processed_docs)

# Filter out infrequent terms appearing less than N times (no_below=N), 
# terms appearing in more than 50% of documents (no_above=0.5), and keep 
# only the top 100,000 terms (keep_n=100000)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Convert dictionary to document – bag of words matrix
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs] #list of lists

#convert the bag of words list of lists to a sparse matrix
term_doc_matrix = gensim.matutils.corpus2csc(bow_corpus)
doc_term_matrix = term_doc_matrix.transpose()
print(doc_term_matrix.shape)

#Display the Bag of Word Matrix
df = pd.DataFrame(doc_term_matrix.toarray().astype('int32'),columns=dictionary.values())
df.head(10)


# In[22]:


# i. Is it sparse or dense?

#SPARSE


# In[23]:


#ii. Find two non-zero entries and briefly interpret their meaning, in words

# Avail/Great/Point/World all have 1 point in doc 1 meaning these listed words showed up 
# the amount of times equal to the corresponding digit in this case 1


# In[24]:


#c. Using logistic regression, partition the data (60% training, 40% validation), and develop a
#model to classify the documents as spam or ham. Comment on its performance.


# In[25]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[26]:


transformer = TfidfTransformer()
matrix = transformer.fit_transform(df) #transform word bag matrix
tfidf = pd.DataFrame(matrix.toarray(), columns = dictionary.values())


# In[28]:


# train test split
from sklearn.model_selection import train_test_split
data = pd.read_csv('SPAM text message.csv')
data['spam'] = data.Category.map({'ham':0, 'spam':1})
X_train, X_test, Y_train, Y_test = train_test_split(tfidf, data['spam'], test_size=0.4, random_state=69420)


# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LogRegression = LogisticRegression()

LogRegression.fit(X_train, Y_train)
prediction = LogRegression.predict(X_test)

tn, fp, fn, tp = confusion_matrix(Y_test,prediction).ravel()
accuracy=round((tn+tp)/(tn+fp+fn+tp),3)
precision=round((tp)/(fp+tp),3)
recall=round((tp)/(tp+fn),3)
print(f"Accuracy is :{accuracy*100}%")
print(f"Precision is :{precision*100}%")
print(f"Recall is :{recall*100}%")


# In[ ]:


# the corresponding recall (73.5%) means the model has detected spam 73.5% 
# furthermore that spam is 96% accurate in being spam

