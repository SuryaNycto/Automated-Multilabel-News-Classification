#edited

import pandas as pd
import numpy as np
from zipfile import ZipFile
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.linear_model import SGDClassifier
import logging
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataset=pd.read_json('News_Category_Dataset_v2.json',lines=True)
dataset['text'] = dataset['headline'] + " " + dataset['short_description']

dataset.isnull().sum()
dataset=dataset[['text']]

nt=pd.read_csv('News Test.csv')

stop_words_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

my_sw = ['make', 'amp',  'news','new' ,'time', 'u','s', 'photos',  'get', 'say']
def black_txt(token):
    return  token not in stop_words_ and token not in list(string.punctuation)  and len(token)>2 and token not in my_sw
  
  
def clean_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "",text)
  text=re.sub("(\\d|\\W)+"," ",text)    
  clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
  clean_text2 = [word for word in clean_text if black_txt(word)]
  return " ".join(clean_text2)

blob = TextBlob((nt.text[7]))
str(blob.correct())


def polarity_txt(text):
  return TextBlob(text).sentiment[0]

def subj_txt(text):
  return  TextBlob(text).sentiment[1]


def len_text(text):
  if len(text.split())>0:
         return len(set(clean_txt(text).split()))/ len(text.split())
  else:
         return 0
     
nt['polarity'] = nt['text'].apply(polarity_txt)
nt.head(5)

nt['subjectivity'] = nt['text'].apply(subj_txt)
nt.head(5)

nt['len'] = nt['text'].apply(len_text)
nt.head(5)



from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction import DictVectorizer
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, data):
        return [{'pos':  row['polarity'], 'sub': row['subjectivity'],  'len': row['len']} for _, row in data.iterrows()]
    
    
    
pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the text
            ('text', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', TfidfVectorizer( min_df =3, max_df=0.2, max_features=None, 
                    strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
                    ngram_range=(1, 10), use_idf=1,smooth_idf=1,sublinear_tf=1,
                    stop_words = None, preprocessor=clean_txt)),
            ])),

            # Pipeline for pulling metadata features
            ('stats', Pipeline([
                ('selector', ItemSelector(key=['polarity', 'subjectivity', 'len'])),
                ('stats', TextStats()),  # returns a list of dicts
                ('vect', DictVectorizer()),  # list of dicts -> feature matrix
            ])),

        ],

        # weight components in FeatureUnion
        transformer_weights={
            'text': 0.9,
            'stats': 1.5,
        },
    ))
])
            
            
seed = 40
nt = nt[['text', 'polarity', 'subjectivity','len']]
Y =dataset['category']
encoder = LabelEncoder()
y = encoder.fit_transform(Y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=seed, stratify=y)

pipeline.fit(x_train)


train_vec = pipeline.transform(x_train)
test_vec = pipeline.transform(nt)
print("Checking that the number of features in train and test correspond: %s - %s" % (train_vec.shape, test_vec.shape))

clf_sv = LinearSVC(C=1, class_weight='balanced', multi_class='ovr', random_state=40, max_iter=10000) #Support Vector machines
clf_sgd = SGDClassifier(max_iter=200,)



from sklearn.model_selection import cross_val_score

clfs = [clf_sv, clf_sgd]
cv = 3
for clf in clfs:
    scores = cross_val_score(clf,train_vec, y_train, cv=cv, scoring="accuracy" )
    print (scores)
    print (("Mean score: {0:.3f} (+/-{1:.3f})").format(
        np.mean(scores), np.std(scores)))
        
from sklearn.metrics import classification_report
clf_sv.fit(train_vec, y_train )
nt_test1 = clf_sv.predict(test_vec)
list_result =[]
list_result.append(("SVC",accuracy_score(y_pred1, y_pred2)))
clf_sgd.fit(train_vec, y_train )
nt_test2 = clf_sgd.predict(test_vec)
list_result.append(("SGD",accuracy_score(x3y_pred1, x3y_pred2)))


# initializing lists 
test_keys = y 
test_values = Y
test_keys=test_keys.tolist()
test_values =test_values.tolist()  
# Printing original keys-value lists 
print ("Original key list is : " + str(test_keys)) 
print ("Original value list is : " + str(test_values)) 
  
# using naive method 
# to convert lists to dictionary 
res = {} 
for key in test_keys: 
    for value in test_values: 
        res[key] = value 
        test_values.remove(value) 
        break  
  
# Printing resultant dictionary  
print ("Resultant dictionary is : " +  str(res))
nttest1_cat = list(map(res.get, nt_test1))
nttest2_cat = list(map(res.get, nt_test2))


nttest1_cat=pd.DataFrame(nttest1_cat,columns=['nttest1_cat'])
nttest2_cat=pd.DataFrame(nttest2_cat,columns=['nttest2_cat'])
nt=nt.join(nttest1_cat)
nt=nt.join(nttest2_cat)

nt.to_csv('nt.csv')            
