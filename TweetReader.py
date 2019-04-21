import pandas as pd
import re as regex
from csv import reader
from csv import writer
from nltk import word_tokenize

'''
Expects a CSV file containing tweets and emotional annotations

will look for a column named <tweet_column_name> in given file, using sep as the separator character
if strip_mentions is true, will remove @ mentions from tweets before adding it

returns the tweets as a pandas data frame
'''
class TweetReader:
    labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']
        
    def read_tweets(self, file, strip_mentions=True, sep_char=',', tweet_column_name='Tweet'):
        self.data = pd.read_csv(file, sep=sep_char, encoding='utf8').drop('ID', axis=1)
        
        if strip_mentions:
            self.data[tweet_column_name] = self.data[tweet_column_name].replace(to_replace='(@\S*)', value='', regex = True)
            self.data = self.data.replace({0: False, 1: True})
            #pd.options.display.encoding = 'unicode'

        data_df = self.data[tweet_column_name]
        label_df = self.data[self.labels]

        return data_df, label_df

    def save_tweets(self, path):
        self.data.to_pickle(path)

    def load_tweets(self, path, tweet_column_name='Tweet'):
        self.data = pd.read_pickle(path)

        data_df = self.data[tweet_column_name]
        label_df = self.data[self.labels]

        return data_df, label_df

    
if __name__ == '__main__':
    reader = TweetReader()
    reader.read_tweets('2018-E-c-En-train.txt', sep_char='\t')

        
        
        
