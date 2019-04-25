from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import FeatureExtraction as Extractor
from nltk import TweetTokenizer
import SVM
import json
import argparse
import pickle
import emoji
import pandas as pd
from sentiment_dict_reader import SentimentDictReader

from io import StringIO
from csv import writer

def read_source(filepath):
    with open(filepath) as file:
        data = json.load(file)
        tweets = {}

        for tweet in data:
            t = tweet['text']
            stamp = tweet['timestamp'][:7]

            if stamp in tweets:
                tweets[stamp].append(t)
            else:
                tweets[stamp] = [t]

        return tweets

def build_features(tweets):
    df = pd.DataFrame(tweets, columns='TWEETS')

    positive_words = SentimentDictReader("dictionaries/augmented/positive-words_semeval.txt")
    negative_words = SentimentDictReader("dictionaries/augmented/negative-words_semeval.txt")

    more_features = Extractor.FeatureExtractor(df, None, tokenizer=TweetTokenizer(reduce_len=True).tokenize, bag_file=None)

    df["EMOJI"] = df.apply(lambda row: get_emoji(row), axis=1)
    df["HAS_QUESTION_MARK"] = df.apply(lambda row: check_question_mark(row), axis=1)
    df["NUM_WORDS"] = df.apply(lambda row: len(row['TWEET'].split(' ')), axis=1)
    df["NUM_POSITIVE_WORDS"] = df.apply(
        lambda row: len(get_sentence_dict_intersection(row["TWEET"].split(" "), positive_words.words)), axis=1)
    df["NUM_NEGATIVE_WORDS"] = df.apply(
        lambda row: len(get_sentence_dict_intersection(row["TWEET"].split(" "), negative_words.words)), axis=1)
    emoji_encoder = preprocessing.LabelEncoder()
    emoji_encoder.fit(df["EMOJI"])
    df["EMOJI"] = pd.Series(emoji_encoder.transform(df["EMOJI"]))

    df["NUM_POSITIVE_BIGRAMS"] = df.apply(
        lambda row: len(get_sentence_bigram_intersection(row["TWEET"], positive_bigrams.words)), axis=1)
    df["NUM_NEGATIVE_BIGRAMS"] = df.apply(
        lambda row: len(get_sentence_bigram_intersection(row["TWEET"], negative_bigrams.words)), axis=1)

    df["FRAC_LOWER_CASE"] = more_features.feature_fraction_lower()
    df["FRAC_UPPER_CASE"] = more_features.feature_fraction_upper()
    df["FRAC_TITLED"] = more_features.feature_fraction_titled()
            
    return df

def process_tweets(data, clf):
    predictions = clf.predict(data)

    num_pos = 0
    num_neg = 0

    for pred in predictions:
        if predictions[pred]:
            num_pos += 1
        else:
            num_neg += 1

    return num_pos, num_neg

def main():
    svm = True
    dec = True
    mlp = True

    tweets = read_source('tweet_data/trump100.json')

    if svm:
        with open('svm.clf', 'rb') as file:
            svm_clf = pickle.load(file)

    if dec:
        with open('decision.clf', 'rb') as file:
            dec_clf = pickle.load(file)

    if mlp:
        with open('mlp.clf', 'rb') as file:
            mlp_clf = pickle.load(file)
        
    
    for month in tweets:
        features = build_features(tweets[month])
        
        total_tweets = len(tweets[month])
        
        if svm:
            svm_pos, svm_neg = process_tweets(features, svm_clf)

        if dec:
            dec_pos, dec_neg = process_tweets(features, dec_clf)

        if mlp:
            mlp_pos, mlp_neg = process_tweets(features, mlp_clf)

        print(month + ' positive %: ', '\n\tsvm: ', str(svm_pos / total_tweets), '\n\tdec: ', str(dec_pos / total_tweets), '\n\tmlp: ', str(mlp_pos / total_tweets))

if __name__ == '__main__':
    main()
