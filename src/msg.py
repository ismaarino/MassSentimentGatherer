######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: msg.py                             #
#                         Autor: Ismael Curto                        #
######################################################################
import globals as g
from tools import warn, jump
from lithops import FunctionExecutor
import json, gathering, processing, queries

def create_fexec(lithops_config):
	try:
		return FunctionExecutor(config=lithops_config)
	except:
		jump(2)
		warn("Unable to create Lithops Function Executor with memory param. Trying default...\n\n")
		return FunctionExecutor(config=lithops_config)


def init():

	config = json.load(open(g.CONFIG_FILE)) # Carreguem la config del fitxer de configuració

	fexec = create_fexec(config["lithops"]) # Creem una instància del Function Executor amb la config del fitxer

	### Part 1 Gathering #########################################################
	#raw_cloud_object = gathering.gather(config, fexec) # Obtenir i guardar dades amb el mòdul gathering
	#print(raw_cloud_object)
	##############################################################################


	### Part 2 Processing ########################################################
	#proc_cloud_object = processing.process(raw_cloud_object, config, fexec) # Processem les dades que hem generat i li apliquem el sentiment analisis
	#print(proc_cloud_object)
	##############################################################################


	### Part 3 Querying ##########################################################
	#print(queries.perform(proc_cloud_object, "SA < 0.5", config, fexec)) # Query d'exemple
	##############################################################################


	fexec.clean() # Llimpiem les dades d'execució del Function FunctionExecutor


if __name__ == "__main__":
	init()
