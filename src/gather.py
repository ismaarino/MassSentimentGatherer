import twint
import csv
import pandas as pd
import re
import time
import io
import copy
import globals as g
from lithops import FunctionExecutor
from store import CloudObjectPublisher
from tfsentiment import Analizer
import datetime


def store_in_csv(csv_writer, t_id, t_time, t_place, t_text, t_likes, analizer):
	try:
		t_text = re.sub(r'\s+', ' ', t_text)
		t_text = bytes(t_text, 'utf-8').decode('utf-8','ignore')
		csv_writer.writerow([t_id, str(t_time), t_text, t_place, 0.0, t_likes])#analizer.analysis(t_text)
	except Exception as e:
		print(str(e)+"\n\n\nCouldn't store tweet "+t_id)

def clf_cos(string):
	r_text = re.sub(r'\s+', ' ', string)
	r_text = re.sub('[^'+g.COS_DELIM+']', ' ', r_text)
	return bytes(r_text, 'utf-8').decode('utf-8','ignore')

#def run_search(config, publisher, analizer):
def run_search(config, publisher):
	tweets = []
	config.Store_object_tweets_list = tweets
	s = twint.run.Search(config)
	#publisher.add_array(tweets)
	for tweet in tweets:
		pass
		#print("Tweet found searching for "+config.Since+" - - - "+config.Until)
		#print(tweet.tweet)
		#publisher.add(clf_cos(tweet.id_str)+g.COS_DELIM+clf_cos(tweet.datestamp)+g.COS_DELIM+clf_cos(tweet.near)+g.COS_DELIM+clf_cos(t_text)+g.COS_DELIM+clf_cos(tweet.likes_count))
		#store_in_csv(csvW, tweet.id_str, tweet.datestamp, tweet.near, tweet.tweet, tweet.likes_count, analizer)
	return len(tweets)

def runtest():
	return 123

def generate_search(langs, topics, places, since, tweets_per_day, threads, conf):
	csvW = csv.writer(io.open(g.CSV_FILE, "w", encoding="utf-8"))
	object_publisher = CloudObjectPublisher(g.COS_BUCKET_NAME)
	d0 = datetime.datetime.strptime(since, '%Y-%m-%d')
	d1 = datetime.datetime.now()
	delta = d1 - d0
	if delta.days <= 0 or delta.days < threads :
		raise Exception("Invalid thread/time config")
	day_incr = delta.days/threads
	limit = day_incr*tweets_per_day
	topic_str = ' '.join([str(topic) for topic in topics])
	si = d0
	with FunctionExecutor(config=conf['lithops']) as fexec:
		#fexec.call_async(object_publisher.start,[])
		c = []
		analizers = {}
		for lang in langs:
			pass#analizers[lang] = Analizer(lang, conf['sentiment_analizer'])
		for t in range(threads) :
			un = si + datetime.timedelta(days=day_incr-1)
			print("Generating search for "+str(si.date())+"  untill "+str(un.date()))
			for lang in langs:
				for place in places:
					c = twint.Config()
					c.Search = topic_str+" lang:"+lang
					c.Lang = lang
					c.Since = str(si.date())+" 00:00:00"
					c.Until = str(un.date())+" 00:00:00"
					c.Near = place
					c.Count = True
					c.Stats = True
					c.Hide_output = True
					c.Store_object = True
					c.Limit = limit

					#fexec.call_async(run_search, [c, object_publisher, analizers[lang]])
					fexec.call_async(run_search, (copy.deepcopy(c), object_publisher))
					#fexec.call_async(runtest, data=())
					print("Tweets:"+str(fexec.get_result()))
					#run_search(c, object_publisher)

			si = si + datetime.timedelta(days=day_incr)




def run(config):
	generate_search(['es'],["COVID"],[""],"2019-12-1",10,1,config)
