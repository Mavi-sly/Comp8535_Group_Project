from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import logging

# clean the text
def clean_post(text):
    text = re.sub(r'https?://\S+', '', text)  # remove URL
    text = re.sub(r'\|\|\|', ' ', text)       # remove split sign
    text = re.sub(r'[^a-zA-Z ]', '', text)    # remove non-alphabetic characters
    text = text.lower()
    return text

# read CSV file and extract data
df = pd.read_csv("mbti_1.csv")
df["cleaned_posts"] = df["posts"].apply(clean_post)

print(df.head())

# num = 0 : 16 classes classification
# num =  1: I or E
#  2: N or S
#  3: T or F
#  4: J or P
def binary(df, num):
    if num == 0 :
        df["label"] = df["type"]
        return df
    if num<0  or num>4:
        print("classification selection error")
        return df
    df["label"]=df["type"].str[num-1]
    return df



# Linear Regression
def logistic(df,num):
    df = binary(df, num)
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df["cleaned_posts"])
    y = df["label"]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(y_pred)
    print(classification_report(y_test, y_pred, target_names=le.classes_))


logistic(df,2)

