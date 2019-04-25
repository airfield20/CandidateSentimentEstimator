from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from TweetReader import TweetReader
import FeatureExtraction as Extractor
from nltk import TweetTokenizer
import argparse
import emoji
import pandas as pd
from sentiment_dict_reader import SentimentDictReader
import nltk

exclude_chars = '#,.\'"`~(): \n*/\\^'


def pos_or_neg(row):
    sum = 0
    if row["anger"]:
        sum = sum - 1
    if row["disgust"]:
        sum = sum - 1
    if row["fear"]:
        sum = sum - 1
    if row["joy"]:
        sum = sum + 1
    if row["love"]:
        sum = sum + 1
    if row["optimism"]:
        sum = sum + 1
    if row["pessimism"]:
        sum = sum - 1
    if row["sadness"]:
        sum = sum - 1
    if row["trust"]:
        sum = sum + 1
    return sum > 0


def get_emoji(row):
    tweet = row["TWEET"]
    emojis = "".join(c for c in tweet if c in emoji.UNICODE_EMOJI)
    if (len(emojis) > 0):
        return emoji.demojize(emojis[0])
    return "none"


def check_question_mark(row):
    for char in row["TWEET"]:
        if char is '?':
            return True
    return False


def get_sentence_dict_intersection(words, dict):
    temp = set(dict)
    lst3 = [value for value in words if value.strip(exclude_chars) in temp]
    return lst3


def get_sentence_bigram_intersection(sentence, bigrams):
    tokens = nltk.word_tokenize(sentence)
    sentence_bigrams = list(nltk.bigrams(tokens))
    bigram_strings = []
    for bigram in sentence_bigrams:
        bigram_strings.append(emoji.demojize(bigram[0].strip(exclude_chars) + ' ' + bigram[1].strip(exclude_chars)).lower())
    temp = set(bigrams)
    lst3 = [value for value in bigram_strings if value in temp]
    return lst3


def get_pos_neg_labels(df):
    return df.apply(lambda row: pos_or_neg(row), axis=1)


def get_unigram_encoder(df):
    unigram_encoder = preprocessing.LabelEncoder()
    words = {}

    def add_to_dict(ls):
        for word in ls:
            words[word.strip(exclude_chars)] = 0

    unigrams = df.apply(lambda row: add_to_dict(row["TWEET"].split(" ")), axis=1)
    unigram_encoder.fit(list(words.keys()))
    return unigram_encoder


def get_unigram_dataframe(df):
    unigrams = {}

    def add_to_dict(ls):
        for word in ls:
            unigrams[word.strip(exclude_chars)] = 0

    df.apply(lambda row: add_to_dict(row["TWEET"].split(" ")), axis=1)
    unigrams = list(unigrams.keys())
    for unigram in unigrams:
        df[unigram] = df.apply(lambda row: 0, axis=1)

    x = 0


def get_most_frequent_pos(df, labels):
    unigrams = {}

    def add_to_dict(ls):
        for word in ls:
            if word in list(unigrams.keys()):
                unigrams[word + ":pos"] = unigrams[word + ":pos"] + 1
            else:
                unigrams[word + ":pos"] = 1

    df.apply(lambda row: add_to_dict(row["TWEET"].split(" ")), axis=1)


def setup():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('data', help='path to the data to be used')
    arg_parser.add_argument('-l', '--load', help='specifies that data is to be loaded from pickled source, not read', action='store_true')
    arg_parser.add_argument('-b', '--bag', help='specifies location of pickled word bags', type=str, default=None)
    classifier_group = arg_parser.add_argument_group('classifiers', 'specify what classifier should be used')
    classifier_group.add_argument('--svm', help='use the svm classifier', action='store_true')
    classifier_group.add_argument('--decision', help='use the decision tree classifier', action='store_true')
    classifier_group.add_argument('--mlp', help='use the mlp classifier', action='store_true')
    classifier_group.add_argument('--all', help='use all classifiers', action='store_true', default=True)

    args = arg_parser.parse_args()

    return (args.data, args.load, args.bag, args.svm, args.decision, args.mlp, args.all)


if __name__ == '__main__':
    data, load_data, bag_file, use_svm, use_decision, use_mlp, use_all = setup()

    if (use_svm or use_decision or use_mlp):
        use_all = False

    reader = TweetReader()

    if load_data:
        docs_df, label_df = reader.load_tweets(data) #'tweet_data/tweet_training.pickle'
    else:
        docs_df, label_df = reader.read_tweets(data, sep_char='\t')

    more_features = Extractor.FeatureExtractor(docs_df, label_df, tokenizer=TweetTokenizer(reduce_len=True).tokenize, bag_file=bag_file)

    ''' FEATURES '''

    df = pd.DataFrame()

    # positive_words = SentimentDictReader("dictionaries/positive-words.txt")
    positive_words = SentimentDictReader("dictionaries/augmented/positive-words_semeval.txt")
    positive_bigrams = SentimentDictReader("dictionaries/dataset_positive_bigrams")
    # negative_words = SentimentDictReader("dictionaries/negative-words.txt")
    negative_words = SentimentDictReader("dictionaries/augmented/negative-words_semeval.txt")
    negative_bigrams = SentimentDictReader("dictionaries/dataset_negative_bigrams")

    df["TWEET"] = docs_df
    # df["IS_POS"] = get_pos_neg_labels(label_df)
    labels = get_pos_neg_labels(label_df)
    df["EMOJI"] = df.apply(lambda row: get_emoji(row), axis=1)
    df["HAS_QUESTION_MARK"] = df.apply(lambda row: check_question_mark(row), axis=1)
    df["NUM_WORDS"] = df.apply(lambda row: len(row['TWEET'].split(' ')), axis=1)
    df["NUM_POSITIVE_WORDS"] = df.apply(
        lambda row: len(get_sentence_dict_intersection(row["TWEET"].split(" "), positive_words.words)), axis=1)
    df["NUM_NEGATIVE_WORDS"] = df.apply(
        lambda row: len(get_sentence_dict_intersection(row["TWEET"].split(" "), negative_words.words)), axis=1)
    df["NUM_POSITIVE_BIGRAMS"] = df.apply(
        lambda row: len(get_sentence_bigram_intersection(row["TWEET"], positive_bigrams.words)), axis=1)
    df["NUM_NEGATIVE_BIGRAMS"] = df.apply(
        lambda row: len(get_sentence_bigram_intersection(row["TWEET"], negative_bigrams.words)), axis=1)

    emoji_encoder = preprocessing.LabelEncoder()
    emoji_encoder.fit(df["EMOJI"])
    df["EMOJI"] = pd.Series(emoji_encoder.transform(df["EMOJI"]))

    df["FRAC_LOWER_CASE"] = more_features.feature_fraction_lower()
    df["FRAC_UPPER_CASE"] = more_features.feature_fraction_upper()
    df["FRAC_TITLED"] = more_features.feature_fraction_titled()

    # get_unigram_dataframe(df)

    ''' TRAINING '''

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25)
    unigram_encoder = get_unigram_encoder(df)
    X_train = X_train.drop(["TWEET"], axis=1)
    X_test = X_test.drop(["TWEET"], axis=1)


    def classify_and_evaluate(classifier, xtrain, ytrain, xtest, ytest):
        classifier.fit(xtrain, ytrain)
        ypred = classifier.predict(xtest)
        f1_score = metrics.f1_score(ytest, ypred)
        accuracy = metrics.accuracy_score(ytest, ypred)
        report = metrics.classification_report(ytest, ypred)

        # Print out results
        print("Percent Accuracy: " + str(accuracy * 100) + "%\n")
        print("F1 score: " + str(f1_score) + "\n")
        # print("Classification report: ")
        # print(str(report))

    if(use_svm or use_all):
        print('-------------SVM CLASSIFIER---------------')
        SVMclassifier = svm.SVC()
        classify_and_evaluate(SVMclassifier, X_train, y_train, X_test, y_test)

    if use_decision or use_all:
        print('-------------Decision Tree CLASSIFIER---------------')
        DTclassifier = DecisionTreeClassifier()
        classify_and_evaluate(DTclassifier, X_train, y_train, X_test, y_test)

    if use_mlp or use_all:
        print('-------------MLP CLASSIFIER---------------')
        MLPclassifier = MLPClassifier()
        classify_and_evaluate(MLPclassifier, X_train, y_train, X_test, y_test)
