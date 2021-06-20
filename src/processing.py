######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: processing.py                      #
#                         Autor: Ismael Curto                        #
######################################################################
import twint
import csv
import pandas as pd
import json
import copy
import globals as g
from storage import CloudDataFramePublisher, CloudRAWReader
import datetime, jsonpickle, pickle
from tools import p, jump, warn, count_elements, normalize


def process_data(thread_data):
	reader = CloudRAWReader(thread_data[0],thread_data[1],"rb")
	lines = reader.readlines(thread_data[2],thread_data[3])
	reader.close()
	from tfsentiment import Analyzer
	Analyzers = {}
	for lang in thread_data[4]["langs"]:
		try:
			Analyzers[lang] = Analyzer(lang, True, thread_data[4])
			Analyzers[lang].setTokenizer(thread_data[5][lang])
		except:
			Analyzers[lang] = 0
			warn("Analyzer for language: "+lang+" couldn't be created.")
	df = pd.DataFrame(columns=thread_data[4]["dataframe_atributes"])
	for pickled in lines:
		unpickled = jsonpickle.decode(pickled)
		for tweet in unpickled:
			tweet_text = normalize(tweet["tweet"])
			try:
				sa = Analyzers[tweet["lang"]].analysis(tweet_text)
			except:
				sa = "-"
				warn("SA for Tweet "+tweet["id_str"]+" couldn't be performed.")
			tweet_dic = {"t_id":tweet["id_str"], "date":tweet["datestamp"], "near":tweet["near"], "content":tweet_text, "likes":tweet["likes_count"], "lang":tweet["lang"], "SA": sa }
			df = df.append(pd.DataFrame([tweet_dic], columns=thread_data[4]["dataframe_atributes"]), ignore_index = False, verify_integrity=False, sort=None)
	return df

def process(cloud_object, conf, fexec):
	p("Counting elements in object: "+cloud_object+" ...")
	lines = count_elements(cloud_object, conf["lithops"], fexec)
	p("The selected raw data object has "+str(lines)+" elements.")
	tokenizers = {}
	for lang in conf["langs"]:
		tokenizers[lang] = copy.deepcopy(pickle.load(open("../models/"+lang+"/"+lang+"_tokenizer.pickle", 'rb')))
	p("Generating thread configurations...")
	i=0
	incr=conf["processing"]["threading_level"]
	while i < lines :
		if i+incr > lines:
			incr = lines-i
		p(str(i)+"-"+str(incr))
		fexec.call_async(process_data, [cloud_object,conf["lithops"], i, incr, conf, tokenizers])
		i+=incr

	results = fexec.get_result()

	nresults = len(results)
	data_cloud_key = ""
	if nresults > 0 :
		p("Completed all with "+str(nresults)+" results.\nStoring results...")
		publisher = CloudDataFramePublisher("data_", conf["processing"]["cloud_file_extension"], conf['lithops'])
		data_cloud_key = publisher.key()
		for result in results:
			if result is not False :
				publisher.commit(result)
		publisher.close()
	else:
		warn("Could not find any data.")
	p("Done.")

	return data_cloud_key
