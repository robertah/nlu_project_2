from config import *
from preprocessing import _load_data
import nltk
import pandas as pd
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentiment_analysis(dataset):
    # load data from csv
    data_original = _load_data(dataset)

    #Only go through the first 10 entries of dataset - Remove for entire dataset
    data_original = data_original.head(10)

    sid = SentimentIntensityAnalyzer()

    sentiment_score = pd.DataFrame(columns=['Compound', 'Negative', 'Neutral', 'Positive'])

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
    return sentiment_score

if __name__ == '__main__':
    print(sentiment_analysis(train_set))

