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

def read_tweets(file, strip_mentions=True, sep_char=',', tweet_column_name='Tweet'):
    data = pd.read_csv(file, sep=sep_char, encoding='utf8')
    if strip_mentions:
        data[tweet_column_name] = data[tweet_column_name].replace(to_replace='(@\S*)', value='', regex = True)
        data = data.replace({0: False, 1: True})
        #pd.options.display.encoding = 'unicode'

        print(data)
    
if __name__ == '__main__':
    read_tweets('2018-E-c-En-train.txt', sep_char='\t')

        
        
        
