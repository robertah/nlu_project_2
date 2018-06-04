from config import *
from preprocessing import load_data
import nltk
import pandas as pd
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentiment_analysis(dataset):

    nltk.download('vader_lexicon')

    # load data from csv
    data_original = load_data(dataset)

    #Only go through the first 10 entries of dataset - Remove for entire dataset
    # data_original = data_original.head(10)

    sid = SentimentIntensityAnalyzer()

    sentiment_score = pd.DataFrame(columns=['compound', 'neg', 'neu', 'pos'])

    #iterate through dataframe for sentiment analysis
    for index, row in data_original.iterrows():

        story_to_complete = " ".join([row['sen1'], row['sen2'], row['sen3'], row['sen4']])
        story_to_complete = "'''{0}'''".format(story_to_complete)
        print(story_to_complete)
        scores = sid.polarity_scores(story_to_complete)
        for key in sorted(scores):
            # print('{0}:{1}, '.format(key, scores[key]), end='')
            print(scores[key])
            sentiment_score.loc[index] = scores

    with open(sentiment_pkl, 'wb') as output:
        pickle.dump(sentiment_score, output, pickle.HIGHEST_PROTOCOL)
        print("Sentiment analysis saved as pkl")

    return sentiment_score


def load_sentiment():

    print("Loading sentiment analysis... ")

    try:
        with open(sentiment_pkl, 'rb') as handle:
            sentiment = pickle.load(handle)
        print("Sentiment analysis loaded")
    except FileNotFoundError:
        print("Sentiment analysis not found.")

    return sentiment



if __name__ == '__main__':
    print(sentiment_analysis(train_set))

