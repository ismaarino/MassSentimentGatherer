import tweepy
import csv
import pandas as pd
import globals as g
import sentiment

def new_tweepy_agent():
	auth = tweepy.OAuthHandler(g.CONSUMER_API_KEY, g.CONSUMER_API_SECRET_KEY)
	auth.set_access_token(g.ACCESS_TOKEN, g.ACCESS_TOKEN_SECRET)
	return tweepy.API(auth)

def store_in_csv(csv_writer,tweet,analizer):
	text = tweet.text.encode('utf-8')
	coords = ""
	if tweet.coordinates != None :
		coords = str(tweet.coordinates["coordinates"][0])+"_"+str(tweet.coordinates["coordinates"][1])
	csv_writer.writerow([tweet.source_url, tweet.created_at, text, coords, analizer.analysis(text), tweet.favorite_count])

def crawl():
	api = new_tweepy_agent()
	csvW = csv.writer(open(g.CSV_FILE, "a"))

	analizer = sentiment.SentimentAnalizer()

	places = api.geo_search(query="Spain",granularity="country")
	country_id = str(places[0].id)

	for hashtag in g.TARGET_HASHTAGS:
		for lang in g.TARGET_LANGS:
			for tweet in tweepy.Cursor(api.search,q=hashtag+" -filter:retweets place:"+country_id,count=100,lang=lang,since="2019-12-1").items():
					store_in_csv(csvW,tweet,analizer)




crawl()