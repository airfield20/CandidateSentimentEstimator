import pandas as pd
import numpy as np
from csv import reader
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

'''
feature extractors

- each function in this file should return a Series or Dataframe of size n_samples x n_features
- when training, we can append all these together selectively, to obtain the final feature set

'''

labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

# finds the most asscoiated words with a particular emotion #
def create_word_bags(documents, labels, classes, keep_highest=4, minimum_score=0.1):
    vec = TfidfVectorizer(stop_words='english')

    tfidf_whole = vec.fit_transform(documents)
    words = vec.get_feature_names()

    bags = {}

    for cl in classes:
        label = labels[cl]
        samples = tfidf_whole[np.where(label)].toarray()

        sample_means = np.mean(samples, axis=0)
        sorted_means_idx = np.argsort(sample_means)[::-1]
        sample_means = [(words[i], sample_means[i]) for i in sorted_means_idx]

        bag = pd.DataFrame(sample_means)
        bag.columns = ['word', 'tfidf_score']

        bags[cl] = set(bag.head(keep_highest)['word'])

    return bags
        
if __name__ == '__main__':
    from TweetReader import read_tweets
    docs_df, label_df = read_tweets('2018-E-c-En-train.txt', sep_char='\t')
    bags = create_word_bags(docs_df, label_df, np.unique(label_df.columns), 60)

    #extractor_out = open('Extraction.out', mode='w')

    for bag in bags:
        print('Label:', bag, '\n', bags[bag], end='\n\n')
        
