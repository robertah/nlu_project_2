from config import *
from preprocessing import load_data
import nltk
import pandas as pd
import numpy as np
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def sentence_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(sentence)
    scores_array = np.asarray(list(scores.values()))
    #print("SCORES ARE ", np.asarray(list(scores.values())))
    return scores_array

def sentiment_analysis(dataset):

    nltk.download('vader_lexicon')

    # load data from csv
    data_original = load_data(dataset)
    print(data_original)
    #Only go through the first 10 entries of dataset - Remove for entire dataset
    # data_original = data_original.head(10)

    sid = SentimentIntensityAnalyzer()

    sentiment_score = pd.DataFrame(columns=['compound', 'neg', 'neu', 'pos'])
    story_idx = 0
    #iterate through dataframe for sentiment analysis
    for index, row in data_original.iterrows():
        #print(row)
        story_to_complete = " ".join([row['sen1'], row['sen2'], row['sen3'], row['sen4']])
        #story_to_complete = "'''{0}'''".format(story_to_complete)
        print(story_to_complete)
        scores = sid.polarity_scores(story_to_complete)
        story_idx = story_idx +1
        if (story_idx%10000 == 0):
            print(story_idx)
        for key in sorted(scores):
            # print('{0}:{1}, '.format(key, scores[key]), end='')
            #print(scores[key])
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
    #Value of sentiment to non negative integer
    #a = np.rint(1000+000*np.asarray(sentiment_analysis(train_set)))#[:,1:3])
    print(sentiment_analysis(train_set))
    #print(np.amax(a))
    #print(np.amin(a))
