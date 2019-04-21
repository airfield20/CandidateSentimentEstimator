from sklearn import svm
from TweetReader import TweetReader
import emoji
import pandas as pd


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
    return sum > -1


def get_emoji(row):
    tweet = row["TWEET"]
    emojis = "".join(c for c in tweet if c in emoji.UNICODE_EMOJI)
    if (len(emojis) > 0):
        return emojis[0]
    return "none"


def check_question_mark(row):
    for char in row["TWEET"]:
        if char is '?':
            return True
    return False


def get_pos_neg_labels(df):
    return df.apply(lambda row: pos_or_neg(row), axis=1)


if __name__ == '__main__':
    reader = TweetReader()
    docs_df, label_df = reader.load_tweets('tweet_data/tweet_training.pickle')
    df = pd.DataFrame()
    positive_words = ['good', 'great', 'love', 'loved', 'superior', 'benefit', 'benefits', 'yay', 'yaay', 'yaaay',
                      'yaaaay', 'happy', 'happiness', 'glee', 'positive', 'positivity', 'ayyye', 'yum', 'enjoy',
                      'enjoyed', 'enjoying','proud',]
    df["TWEET"] = docs_df
    df["IS_POS"] = get_pos_neg_labels(label_df)
    df["EMOJI"] = df.apply(lambda row: get_emoji(row), axis=1)
    df["HAS_QUESTION_MARK"] = df.apply(lambda row: check_question_mark(row), axis=1)
    df["NUM_WORDS"] = df.apply(lambda row: len(row['TWEET'].split(' ')), axis=1)
    x = 10
