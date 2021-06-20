######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: gathering.py                       #
#                         Autor: Ismael Curto                        #
######################################################################
import twint, csv
import pandas as pd
import json, re, time, io, copy
from tools import p, jump, warn
import globals as g
from storage import CloudDataFramePublisher, CloudRawDataPublisher
import datetime, unicodedata, jsonpickle

class Tweet():
	pass

def normalize(data):
	return unicodedata.normalize('NFKC', data)

def run_search(thread_config):
	tweets_original = []
	tweets = []
	thread_config[0].Store_object_tweets_list = tweets_original
	s = twint.run.Search(thread_config[0])
	for tweet in tweets_original:
		tweet_data_object = Tweet()
		for attr in thread_config[1]:
			setattr(tweet_data_object, attr, getattr(tweet, attr))
		setattr(tweet_data_object, "lang", thread_config[0].Lang)
		tweets.append(tweet_data_object)
	return jsonpickle.encode(tweets, unpicklable=False)

def gather(conf, fexec):
	jump(3)
	p("Starting job with config:\n\t\tLanguages: "+str(conf["langs"])+",\n\t\tTopics: "+str(conf["search"]["topics"])+",\n\t\tNear: "+str(conf["search"]["places"])+",\n\t\tSince: "+conf["search"]["since"]+",\n\t\tTo: "+conf["search"]["to"]+",\n\t\tMax.TPD: "+str(conf["search"]["tweets-per-day"])+"\n")
	p("Generating time intervals... \n")
	d0 = datetime.datetime.strptime(conf["search"]["since"], '%Y-%m-%d')
	if conf["search"]["to"] == "now" :
		d1 = datetime.datetime.now()
	else :
		d1 = datetime.datetime.strptime(conf["search"]["to"], '%Y-%m-%d')
	delta = d1 - d0
	if delta.days <= 0 :
		raise Exception("Invalid time config")
	threadinglevel = delta.days / conf["search"]["threading_level"]
	if threadinglevel < 1 :
		threadinglevel = 1
	day_incr = delta.days / threadinglevel
	si = d0

	configs = []

	c = twint.Config()
	c.Count = True
	c.Stats = True
	c.Hide_output = True
	c.Store_object = True
	c.Limit = day_incr * conf["search"]["tweets-per-day"]

	p("Generating Twint (https://github.com/twintproject/twint) search configurations...")

	for t in range(int(threadinglevel)) :
		un = si + datetime.timedelta(days=day_incr-1)
		p(str(un)+" "+str(si))
		for lang in conf["langs"]:
			for place in conf["search"]["places"]:
				for topic in conf["search"]["topics"]:
					c.Search = topic+" lang:"+lang
					c.Lang = lang
					c.since = str(si.date())+" 00:00:00"
					c.Until = str(un.date())+" 00:00:00"
					c.Near = place

					configs.append([copy.deepcopy(c),conf["wanted_twint_atributes"]])

		si = si + datetime.timedelta(days=day_incr)
		margin = d1 - si
		if margin.days <= 0 :
			break
	p("Generated "+str(len(configs))+" configurations for the specified input parameters.")
	p("Launching Lithops Cloud Function Executor")
	p("Waiting to complete "+str(len(configs))+" threads")
	jump(3)
	fexec.map(run_search, configs)
	results = fexec.get_result()
	#results =  []

	nresults = len(results)
	object_key = ""
	if nresults > 0 and len(results[0]) > 0 :
		p("Completed all with "+str(nresults)+" results.\nStoring results...")
		publisher = CloudRawDataPublisher("raw", conf["search"]["cloud_file_extension"], conf['lithops'])

		for i in range(nresults):
			publisher.commit(results[i])
		object_key = publisher.key()
		publisher.close()
	else:
		warn("Could not find any data.")
	p("Done.")

	return object_key
