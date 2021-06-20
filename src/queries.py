######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: queries.py                         #
#                         Autor: Ismael Curto                        #
######################################################################
import fsspec
import pandas as pd
from tools import p, jump, warn, count_elements
from storage import CloudReader, CloudDFReader

def join_results(results):
	try:
		joined = 0
		if results[0].isnumeric() :
			for result in results:
				try:
					joined += result
				except:
					pass
			return joined
	except:
		pass
	joined = pd.DataFrame(columns=["t_id","date","near","content","likes","lang","SA"])
	for result in results:
		joined = joined.append(result)
	return joined

def partial_query(thread_data):
	cloud_object = thread_data[0]
	lithops_config = thread_data[1]
	index = thread_data[2]
	length = thread_data[3]
	query = thread_data[4]
	df = CloudDFReader(cloud_object, lithops_config,"rb").dataframe(index, length)
	return df.query(query)

def count(dataframe):
	return len(dataframe.index)


def perform(cloud_object, query, conf, fexec):
	p("Counting elements in object: "+cloud_object+" ...")
	lines = count_elements(cloud_object, conf["lithops"], fexec)
	p("The selected raw data object has "+str(lines)+" elements.")
	if lines <= 0:
		raise Exception("Cloud object "+cloud_object+" does not exist or is empty.")
	p("Obtaining results...")
	processing_list = []
	incr = int(lines / conf["queries"]["threading_level"] + 1)
	i = 1
	while i < lines :
		if i+incr > lines:
			incr = lines-i
		p("Perparing "+str(i)+" "+str(incr))
		processing_list.append([cloud_object, conf["lithops"], i, incr, query])
		i = i+incr

	fexec.map(partial_query, processing_list)
	results = fexec.get_result()
	p("Joining Results at Cloud...")
	fexec.call_async(join_results, results)
	unified = fexec.get_result()

	if len(unified) <= 0 :
		warn("Could not find any data.")
	print("Done.")
	return unified
