# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:28:58 2020

@author: Henok Gashaw
"""
# SMS "SPAM/HAM" CLASSIFIER PROJECT CODE

# Importing the Dataset

import pandas as pd

messages = pd.read_csv('spam.tsv', sep = '\t')

# Data cleaning and preprocessing

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
wl = WordNetLemmatizer()

corpus = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [wl.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

corpus = pd.DataFrame(corpus, columns = ['message'] )
corpus.insert(1, 'label', messages['label'])

# Balancing the dataset

ham = corpus[corpus['label'] == 'ham']
spam = corpus[corpus['label'] == 'spam'] 
ham = ham.sample(spam.shape[0])

data = ham.append(spam, ignore_index = True)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size = 0.25, random_state = 0, shuffle = True, stratify = data['label'])

# Feature vector formation

# Creating  Bag of words and TF-IDF vectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
cv = CountVectorizer(max_features = 2500)
#tf_vec = TfidfVectorizer()

X_train = cv.fit_transform(X_train)
#X_train = tf_vec.fit_transform(X_train)

# Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
NB_classifier = Pipeline([('cv', CountVectorizer()), ('clf', MultinomialNB())])
#NB_classifier = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])

# Training

NB_classifier.fit(X_train, y_train)

# Prediction

NB_prediction = NB_classifier.predict(X_test)

# Performance evaluation

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
confusion_matrix_NB = confusion_matrix(NB_prediction, y_test)
accuracy_NB = accuracy_score(NB_prediction, y_test)*100
classification_report_NB = classification_report(NB_prediction, y_test)

# Save a model

import joblib
joblib.dump(NB_classifier, 'NB_classifierModel.pkl')

# x = NB_classifier.predict([' send OK to 6787 for free'])
# print(x)
# y = NB_classifier.predict(['Hello mates. we have defence tommorow at 8 AM. Good luck'])
# print(y)
# while True:
#     print('Please enter the text(type ''q'' to quit): ')
#     raw_input = input()
#     if raw_input == 'q':
#         break
#     answer = NB_classifier.predict([raw_input])
#     print(answer)
