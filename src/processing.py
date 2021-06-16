import twint
import csv
import pandas as pd
import json
import copy
import globals as g
from lithops import FunctionExecutor
from store import CloudDataFramePublisher, CloudReader
import datetime
import unicodedata
import jsonpickle
from tfsentiment import Analizer

def normalize(data):
	return unicodedata.normalize('NFKC', data)

def process_data(thread_data):
	pickled_arr = thread_data[0]
	analizers = thread_data[1]
	df = pd.DataFrame(columns=["t_id","date","near","content","likes","lang","SA"])
	for pickled in pickled_arr:
		unpickled = jsonpickle.decode(pickled)
		for tweet in unpickled:
			tweet_text = normalize(tweet["tweet"])
			tweet_dic = {"t_id":tweet["id_str"], "date":tweet["datestamp"], "near":tweet["near"], "content":tweet_text, "likes":tweet["likes_count"], "lang":tweet["lang"], "SA":analizers[tweet["lang"]].analysis(tweet_text) }
			df = df.append(pd.DataFrame([tweet_dic], columns=["t_id","date","near","content","likes"]), ignore_index = False, verify_integrity=False, sort=None)
	return df

def process(cloud_object, conf):
	analizers = {}
	for lang in conf["langs"]:
		analizers[lang] = Analizer(lang, conf)
	pickled_arr = CloudReader(cloud_object, conf["lithops"]).readlines()
	print("Downloaded object with "+str(len(pickled_arr))+" elements.")
	processing_list = []
	tl = conf["processing"]["threading_level"]
	increment = int(len(pickled_arr) / tl)
	if increment == 0:
		increment = 1
	i = 0
	while i < len(pickled_arr) :
		t_list = []
		for j in range(i,i+increment):
			if j >= len(pickled_arr):
				break
			t_list.append(pickled_arr[j])
			processing_list.append([copy.deepcopy(t_list), copy.deepcopy(analizers)])
		i += increment
	fexec = FunctionExecutor(config=conf["lithops"])
	fexec.map(process_data, processing_list)
	results = fexec.get_result()
	print(results[0])
	nresults = len(results)
	if nresults > 0 and len(results[0]) > 0 :
		print("\n\n"+g.HEAD+"Completed all with "+str(nresults)+" results.\nStoring results...")
		publisher = CloudDataFramePublisher("data", conf["processing"]["cloud_file_extension"], conf['lithops'])
		for i in range(nresults):
			publisher.commit(results[i])
		publisher.close()
		fexec.clean()
	else:
		print(g.HEAD+"Could not find any data.")
	print("Done.")
