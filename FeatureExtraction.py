import pandas as pd
import numpy as np
import string
import pickle
from csv import writer
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TweetTokenizer

class FeatureExtractor:
    def __init__(self, documents, labels, tokenizer=None, stop_words=None, bag_file=None):
        self.documents = documents
        self.labels = labels
        self.classes = labels.columns
        self.pos_labels = ['joy', 'love', 'optimism', 'trust']
        self.neg_labels = ['anger', 'disgust', 'fear', 'pessimism', 'sadness']
        
        if not stop_words:
            self.custom_stopwords = stop = set([word for word in ENGLISH_STOP_WORDS]).union(set(stopwords.words('English'))).union(set(['\\', ':' '/', '-', "'", '"', '.', ',', "i've", "i'm", "i'll", '&']))
        else:
            self.custom_stopwords = stop_words

        self.tokenizer = word_tokenize if not tokenizer else tokenizer

        if bag_file:
            try:
                with open(bag_file, 'rb') as file:
                    self.bags = pickle.load(file)
                    self.reduce_bag_to_pos_neg()
                    self.set_count = len(self.bags[self.classes[0]])
            except IOError:
                print("An error occured while trying to read file.")
                self.bags = None
        else:
            self.bags = None
    
# finds the most asscoiated words with a particular emotion #
    def create_word_bags(self, keep_highest=20):
        self.set_count = keep_highest
        vec = TfidfVectorizer(stop_words=self.custom_stopwords, tokenizer=self.tokenizer)

        tfidf_whole = vec.fit_transform(self.documents)
        words = vec.get_feature_names()

        self.bags = {}

        for cl in self.classes:
            label = self.labels[cl]
            samples = tfidf_whole[np.where(label)].toarray()

            sample_means = np.mean(samples, axis=0)
            sorted_means_idx = np.argsort(sample_means)[::-1]
            sample_means = [(words[i], sample_means[i]) for i in sorted_means_idx]

            bag = pd.DataFrame(sample_means)
            bag.columns = ['word', 'tfidf_score']

            self.bags[cl] = set(bag.head(keep_highest)['word'])

        self.reduce_bag_to_pos_neg()

    def save_word_bags(self):
        with open('extracted-bags-count-' + str(self.set_count) + '.pickle', 'wb') as file:
            pickle.dump(self.bags, file)

    def reduce_bag_to_pos_neg(self):
        # assert not self.bags.empty(), "No word bag found. Did you call create_word_bags() before this?"

        self.positive = set()
        self.negative = set()

        for bag in self.bags:
            if bag in self.pos_labels:
                self.positive = self.positive.union(self.bags[bag])
            elif bag in self.neg_labels:
                self.negative = self.negative.union(self.bags[bag])

    def feature_count_class_words(self):
        assert self.bags, "No word bag found. Did you call create_word_bags() before this?"
            
        features_csv = StringIO()
        feature_writer = writer(features_csv)

        feature_writer.writerow(self.classes)
        
        for idx, document in self.documents.iteritems():
            row = pd.Series(index=self.classes, dtype=np.int32).fillna(value=0)
            tokens = self.tokenizer(document)
            for token in tokens:
                for cls in self.classes:
                    if token in self.bags[cls]:
                        row[cls] += 1

            feature_writer.writerow(row.tolist())

        features_csv.seek(0)

        return pd.read_csv(features_csv, engine='c')

    def feature_count_pos_neg(self):
        assert self.positive, "No positive word bag found. Did you call reduce_bag_to_pos_neg() before this?"

        features_csv = StringIO()
        feature_writer = writer(features_csv)

        feature_writer.writerow(['positive', 'negative'])

        for idx, document in self.documents.iteritems():
            positive_count = 0
            negative_count = 0
            
            tokens = self.tokenizer(document)
            for token in tokens:
                if token in self.positive:
                    positive_count += 1
                elif token in self.negative:
                    negative_count += 1

            feature_writer.writerow([positive_count, negative_count])

        features_csv.seek(0)

        return pd.read_csv(features_csv, engine='c')
                
# document length in words (basically, len(tokenizer(text)) #
    def get_doc_len(self, text):
        return len([word for word in self.tokenizer(text) if word not in string.punctuation])

    def feature_doc_len(self):
        return self.documents.apply(self.get_doc_len).to_frame(name='doc_len')

# average word length #
    def get_avg_word_len(self, text):
        words = self.tokenizer(text)
        return sum([len(word) for word in words]) / len(words)
    
    def feature_avg_word_len(self):
        return self.documents.apply(self.get_avg_word_len).to_frame(name='avg_word_len')

# frequency of upper, lower, and title cased words #
    def get_upper_count(self, text):
        return len([word for word in self.tokenizer(text) if word.isupper()])

    def get_lower_count(self, text):
        return len([word for word in self.tokenizer(text) if word.islower()])

    def get_titled_count(self, text):
        return len([word for word in self.tokenizer(text) if word.istitle()])

    def feature_fraction_upper(self):
        return self.documents.apply(lambda text: self.get_upper_count(text) / self.get_doc_len(text)).to_frame(name='fraction_upper')

    def feature_fraction_lower(self):
        return self.documents.apply(lambda text: self.get_lower_count(text) / self.get_doc_len(text)).to_frame(name='fraction_lower')

    def feature_fraction_titled(self):
        return self.documents.apply(lambda text: self.get_titled_count(text) / self.get_doc_len(text)).to_frame(name='fraction_titled')
        
if __name__ == '__main__':
    from TweetReader import TweetReader

    reader = TweetReader()
    docs_df, label_df = reader.load_tweets('tweet_data/tweet_training.pickle')

    token_gen = TweetTokenizer(reduce_len=True)
    extractor = FeatureExtractor(documents=docs_df, labels=label_df, tokenizer=token_gen.tokenize, bag_file='extracted-bags/extracted-bags-count-90.pickle')
    
    extractor.create_word_bags(150)
    extractor.save_word_bags()
    
    print(extractor.feature_count_class_words().head(20), '\n')
    print(extractor.feature_doc_len().head(20), '\n')
    print(extractor.feature_avg_word_len().head(20), '\n')
    print(extractor.feature_fraction_upper().head(20), '\n')
    print(extractor.feature_fraction_lower().head(20), '\n')
    print(extractor.feature_fraction_titled().head(20), '\n')

    

    
        
