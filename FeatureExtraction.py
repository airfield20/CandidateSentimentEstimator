import pandas as pd
import numpy as np
from csv import reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer

# custom stopwords #
ENGLISH_CUSTOM = stop = set([word for word in ENGLISH_STOP_WORDS]).union(set(stopwords.words('English'))).union(set(['\\', ':' '/', '-', "'", '"', '.', ',', "i've", "i'm", "i'll", '&']))

# finds the most asscoiated words with a particular emotion #
def create_word_bags(documents, labels, classes, keep_highest=4, custom_tokenizer=None):
    if custom_tokenizer:
        vec = TfidfVectorizer(stop_words=ENGLISH_CUSTOM, tokenizer=custom_tokenizer)
    else:
        vec = TfidfVectorizer(stop_words=ENGLISH_CUSTOM)

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

    token_gen = TweetTokenizer(reduce_len=True)
    
    bags = create_word_bags(docs_df, label_df, np.unique(label_df.columns), 60, custom_tokenizer=token_gen.tokenize)

    #extractor_out = open('Extraction.out', mode='w')

    for bag in bags:
        print('Label:', bag, '\n', bags[bag], end='\n\n')
        
