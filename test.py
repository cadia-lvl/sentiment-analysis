import re
import math
import nltk
import pandas as pd
import numpy as np
from reynir import Greynir
from googletrans import Translator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer, word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk import stem
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

t = Translator()
g = Greynir()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
nltk.download('punkt')
def google_translate(text):
    text = text.replace("<br />", "")
    return t.translate(text, dest="is").text

def lemmatize_english_text(text):
    st = ""
    text = text.replace("<br />", "")
    return word_tokenize(text)
    # for w in w_tokenizer.tokenize(text):
    #     st = st + lemmatizer.lemmatize(w) + " "
    return st
    

data = pd.read_csv('IMDB-Dataset.csv')
#data_ice_google = pd.read_csv("IMDB-Dataset-GoogleTranslate.csv")
#data_ice_midein = pd.read_csv("IMDB-Dataset-MideindTranslate.csv")

first_review = data['review'][0]
first_review_icelandic = google_translate(first_review)

#print(first_review[0:100])
#print(first_review_icelandic[0:100])
print(lemmatize_english_text(first_review))
print('---')
print(first_review)

# test = g.submit(first_review_icelandic)
# for sent in test:
#     sent.parse()
#     print(sent.tidy_text)
    #print(sent.lemmas)