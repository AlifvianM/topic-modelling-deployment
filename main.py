from typing import Union, List
from fastapi import FastAPI

import pandas as pd
import string
import nltk
import pickle
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = FastAPI()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("indonesian"))
translator = str.maketrans("", "", string.punctuation)
load_bow = pickle.load(open("bow_corpus","rb"))
ldamodel_load = LdaModel.load('ldamodel.model')

def preprocess(text):
    text = text.lower()
    text = text.translate(translator)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

def transform_to_list(dataset):
    clean_text = []
    for text in dataset:
        print(text)
        clean_text.append(preprocess(text))
    return clean_text

def bow_corp(transformed_text):
    dictionary = Dictionary(transformed_text)
    bow_corpus = [dictionary.doc2bow(text) for text in transformed_text]
    return bow_corpus

def predict(bow):
    topic = []
    for i, doc in enumerate(bow):
        doc_topics = ldamodel_load.get_document_topics(doc)
        topic.append(doc_topics)
    return topic

def process_result(res):
    topics_doc = []
    for i in res:
        top = 0
        top_rate = 0
        for j in i:
            if j[1] > top:
                top = j[1]
                top_rate = j[0]
        topics_doc.append(top_rate)

    return topics_doc

@app.get('/')
async def root():
    return "hello world!"


@app.post("/predicts/")
async def predict_text(text: List[str]):
    print("Text :", text)
    clean_texts = transform_to_list(text)
    print("Clean Texts :", clean_texts)
    bow = bow_corp(clean_texts)
    print("BOW :", bow)
    result = process_result(predict(bow))
    print("Result :",result, type(result))
    return {
        "status":200,
        "result":result
    }