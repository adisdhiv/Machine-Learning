import tweepy
from textblob import TextBlob
import csv

# Step 1 - Authenticate
consumer_key= ''
consumer_secret= ''

access_token=''
access_token_secret=''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#Step 3 - Retrieve Tweets
public_tweets = api.search('flood')

#CHALLENGE - Instead of printing out each tweet, save each Tweet to a CSV file
#and label each one as either 'positive' or 'negative', depending on the sentiment 
#You can decide the sentiment polarity threshold yourself
polarity = []
tweets = []

for tweet in public_tweets:
    print(tweet.text) #.text to print string type of it
    
    #Step 4 Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    if analysis.sentiment.polarity > 0.3:
        polarity.append(analysis.sentiment.polarity)
        tweets.append(analysis)
print("")
# polarity - whether it is positive or negative
# subjectivity - whether it is opinion or factual

# print to csv file
with open('tweets.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(zip(tweets, polarity))
csvFile.close()
