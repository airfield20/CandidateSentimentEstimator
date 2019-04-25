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
from csv import writer
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
    df = pd.DataFrame(tweets, columns=["TWEET"])

    positive_words = SentimentDictReader("dictionaries/augmented/positive-words_semeval.txt")
    positive_bigrams = SentimentDictReader("dictionaries/dataset_positive_bigrams")
    negative_words = SentimentDictReader("dictionaries/augmented/negative-words_semeval.txt")
    negative_bigrams = SentimentDictReader("dictionaries/dataset_negative_bigrams")

    more_features = Extractor.FeatureExtractor(df['TWEET'], None, tokenizer=TweetTokenizer(reduce_len=True).tokenize, bag_file=None)

    df["EMOJI"] = df.apply(lambda row: SVM.get_emoji(row), axis=1)
    df["HAS_QUESTION_MARK"] = df.apply(lambda row: SVM.check_question_mark(row), axis=1)
    df["NUM_WORDS"] = df.apply(lambda row: len(row['TWEET'].split(' ')), axis=1)
    df["NUM_POSITIVE_WORDS"] = df.apply(
        lambda row: len(SVM.get_sentence_dict_intersection(row["TWEET"].split(" "), positive_words.words)), axis=1)
    df["NUM_NEGATIVE_WORDS"] = df.apply(
        lambda row: len(SVM.get_sentence_dict_intersection(row["TWEET"].split(" "), negative_words.words)), axis=1)
    emoji_encoder = preprocessing.LabelEncoder()
    emoji_encoder.fit(df["EMOJI"])
    df["EMOJI"] = pd.Series(emoji_encoder.transform(df["EMOJI"]))

    df["NUM_POSITIVE_BIGRAMS"] = df.apply(
        lambda row: len(SVM.get_sentence_bigram_intersection(row["TWEET"], positive_bigrams.words)), axis=1)
    df["NUM_NEGATIVE_BIGRAMS"] = df.apply(
        lambda row: len(SVM.get_sentence_bigram_intersection(row["TWEET"], negative_bigrams.words)), axis=1)

    df["FRAC_LOWER_CASE"] = more_features.feature_fraction_lower()
    df["FRAC_UPPER_CASE"] = more_features.feature_fraction_upper()
    df["FRAC_TITLED"] = more_features.feature_fraction_titled()
            
    return df.drop(['TWEET'], axis=1)

def process_tweets(data, clf):
    predictions = clf.predict(data)

    num_pos = 0
    num_neg = 0

    for pred in predictions:
        if pred:
            num_pos += 1
        else:
            num_neg += 1

    return num_pos, num_neg

def main():
    tweets = read_source('tweet_data/trump10000since2016.json')

    with open("trump10000.csv", 'w') as csv_file:
        
        csv_writer = writer(csv_file)

        csv_writer.writerow(['month', 'svm positive', 'svm negative', 'decision tree positive', 'decision tree negative', 'mlp positive', 'mlp negative'])
    
        with open('svm.clf', 'rb') as file:
            svm_clf = pickle.load(file)


        with open('decision.clf', 'rb') as file:
            dec_clf = pickle.load(file)

        
        with open('mlp.clf', 'rb') as file:
            mlp_clf = pickle.load(file)
            
        
        for month in tweets:
            features = build_features(tweets[month])
            
            total_tweets = len(tweets[month])
            
            
            svm_pos, svm_neg = process_tweets(features, svm_clf)

            
            dec_pos, dec_neg = process_tweets(features, dec_clf)

            
            mlp_pos, mlp_neg = process_tweets(features, mlp_clf)

            csv_writer.writerow([month, str((svm_pos / total_tweets) * 100), str((svm_neg / total_tweets) * 100), str((dec_pos / total_tweets) * 100), str((dec_neg / total_tweets) * 100), str((mlp_pos / total_tweets) * 100), str((mlp_neg / total_tweets) * 100)])

if __name__ == '__main__':
    main()
