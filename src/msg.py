import json
import globals as g
import gathering, processing


def init():
	### Part 1 #######################################
	config = json.load(open(g.CONFIG_FILE)) # Carreguem la config del fitxer de configuració
	#cloud_object = gathering.gather(config) # Executem la funció per obtenir i guardar dades del mòdul gathering
	#print(cloud_object)
	### Part 2 #######################################
	cloud_object = "raw_Jun-17-2021____6995e5aa-cef8-11eb-b923-e4e749360e9f.rawdata"
	processing.process(cloud_object, config) # Processem les dades que hem generat i li apliquem el sentiment analisis

	### Part 3 #######################################

init()
