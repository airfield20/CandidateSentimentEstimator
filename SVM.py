from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from TweetReader import TweetReader
import emoji
import pandas as pd
from sentiment_dict_reader import SentimentDictReader

exclude_chars = '#,.\'"`~(): '


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


if __name__ == '__main__':
    reader = TweetReader()
    docs_df, label_df = reader.load_tweets('tweet_data/tweet_training.pickle')
    df = pd.DataFrame()
    # positive_words = SentimentDictReader("dictionaries/positive-words.txt")
    positive_words = SentimentDictReader("dictionaries/augmented/positive-words_semeval.txt")
    # negative_words = SentimentDictReader("dictionaries/negative-words.txt")
    negative_words = SentimentDictReader("dictionaries/augmented/negative-words_semeval.txt")
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
    emoji_encoder = preprocessing.LabelEncoder()
    emoji_encoder.fit(df["EMOJI"])
    df["EMOJI"] = pd.Series(emoji_encoder.transform(df["EMOJI"]))
    # get_unigram_dataframe(df)

    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.25)
    unigram_encoder = get_unigram_encoder(df)
    X_train = X_train.drop(["TWEET"], axis=1)
    X_test = X_test.drop(["TWEET"], axis=1)

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)

    # classifier = DecisionTreeClassifier()
    # classifier.fit(X_train, y_train)

    # classifier = MLPClassifier()
    # classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    f1_score = metrics.f1_score(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    # Print out results
    print("Percent Accuracy: " + str(accuracy * 100) + "%\n")
    print("F1 score: " + str(f1_score) + "\n")
    print("Classification report: ")
    print(str(report))

    x = 10
