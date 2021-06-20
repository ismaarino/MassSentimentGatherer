######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: tools.py                           #
#                         Autor: Ismael Curto                        #
######################################################################
import globals as g
import sys, unicodedata, storage
from lithops import FunctionExecutor

def p(text):
	print(g.HEAD+str(text))

def warn(text):
	print(g.WHEAD+str(text))

def jump(n):
	for i in range(n):
		print(" ")

def normalize(data):
	return unicodedata.normalize('NFKC', data)

def bytesinstringarray(data):
	sum = 0
	for e in data:
		sum += len(e.encode('utf-8'))
	return sum

def count_lines(data):
	return storage.CloudReader(data[0], data[1], "rb").len()

def count_elements(cloud_object, lithops_config, fexec):
	lines = 0
	try:
		p("Attemting to remotely count elements...")
		fexec.call_async(count_lines, [cloud_object, lithops_config])
		lines = fexec.get_result()
	except:
		warn("Remote element count failed, object probably is too big for purchased cloud license. Doing locally...")
		lines = count_lines([cloud_object, lithops_config])
	if lines <= 0:
		raise Exception("Cloud object "+cloud_object+" does not exist or is empty.")
	return lines
