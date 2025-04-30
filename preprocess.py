import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
mbti_df = pd.read_csv('mbti_1.csv') # read the csv file

# 1. Text normalization 
# (i) convert to lowercase;
# (ii) remove url; 
# (iii) remove numbers;
# (iv) remove non-alphanumeric characters (punctuation, special characters);
# (v) remove underscores and signs;
# (vi) replace multiple spaces with single spaces;
# (vii) remove stopwords; 
# (viii) remove one-letter words;
def clean_text(text):
    text = str(text)
    text = text.lower()
    pattern = re.compile(r'https?://[a-zA-Z0-9./-]*/[a-zA-Z0-9?=_.]*[_0-9.a-zA-Z/-]*')
    text = re.sub(pattern, ' ', text)
    pattern = re.compile(r'[0-9]')
    text = re.sub(pattern, ' ', text)
    pattern = re.compile(r'\W+')
    text = re.sub(pattern, ' ', text)
    pattern = re.compile(r'[_+]')
    text = re.sub(pattern, ' ', text)
    pattern = re.compile(r'\s+')
    text = re.sub(pattern, ' ', text).strip()
    stop_words = stopwords.words("english")
    text = " ".join([w for w in text.split() if w not in stop_words])
    text = ' '.join([word for word in text.split() if len(word) > 1])
    return text

# 2. Lemmatization
# (i) use NLTK's lemmatizer
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# 3. Select classification dimension (16-class classification or binary classification on single dimension)
def select_classification_dimension(df, num=0):
    if num == 16:
        return df['type']
    elif 1 <= num <= 4:
        return df['type'].str[num-1]
    else:
        print("selection error of classification dimension!")
        return df['type']

# 4. Label encoding
# (i) creates an array corresponding to the type labels.
def encode_labels(column):
    le = LabelEncoder()
    y = le.fit_transform(column)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"Label encoding mapping: {mapping}")
    print(f"Encoded label examples: {y[:5]}")
    print(f"Unique encoded labels: {np.unique(y)}")
    return y, le

# 1. apply text normalization to 'posts' column
mbti_df['posts'] = mbti_df['posts'].apply(clean_text)

# 2. apply lemmatization to 'posts' column
mbti_df['posts'] = mbti_df['posts'].apply(lemmatize_text)

# 16: 16 classes classification; 
# 1: classify only I or E; 
# 2: classify only N or S; 
# 3: classify only T or F; 
# 4: classify only J or P.
classification_mode = 2

# 3. obtain the corresponding labels for the classification dimension
selected_labels = select_classification_dimension(mbti_df, classification_mode)

# 4. apply label encoding to 'type' column
y,le = encode_labels(selected_labels)

# 5. Use TF-IDF vectorization to extract features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(mbti_df['posts']).toarray()

# 6. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. The following is a test for the effectiveness of preprocessing using Logistic Regression
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# The following is the preprocessing code of last version
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import string
# import numpy as np
# import nltk
# import matplotlib.pyplot as plt
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# import logging

# # clean the text
# def clean_post(text):
#     text = re.sub(r'https?://\S+', '', text)  # remove URL
#     text = re.sub(r'\|\|\|', ' ', text)       # remove split sign
#     text = re.sub(r'[^a-zA-Z ]', '', text)    # remove non-alphabetic characters
#     text = text.lower()
#     return text

# # read CSV file and extract data
# df = pd.read_csv("mbti_1.csv")
# df["cleaned_posts"] = df["posts"].apply(clean_post)

# print(df.head())

# # num = 0 : 16 classes classification
# # num =  1: I or E
# #  2: N or S
# #  3: T or F
# #  4: J or P
# def binary(df, num):
#     if num == 0 :
#         df["label"] = df["type"]
#         return df
#     if num<0  or num>4:
#         print("classification selection error")
#         return df
#     df["label"]=df["type"].str[num-1]
#     return df



# # Linear Regression
# def logistic(df,num):
#     df = binary(df, num)
#     tfidf = TfidfVectorizer(max_features=5000)
#     X = tfidf.fit_transform(df["cleaned_posts"])
#     y = df["label"]

#     le = LabelEncoder()
#     df["label"] = le.fit_transform(df["label"])

#     X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

#     clf = LogisticRegression(max_iter=2000)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     print(y_pred)
#     print(classification_report(y_test, y_pred, target_names=le.classes_))


# logistic(df,2)
