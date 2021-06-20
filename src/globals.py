######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: globals.py                         #
#                         Autor: Ismael Curto                        #
######################################################################

# MSG config File
CONFIG_FILE = "../config/msgconf.json"

# MSG samples dirs
SAMPLES_DIR = "../training/"
SENTIMENTS_DIR = SAMPLES_DIR+"sentiments/"
WORDVECTORS_DIR = SAMPLES_DIR+"wordvectors/"


PUB_INTERVAL = 1

# MSG samples Folder
MODELS_DIR = "../models/"

# Wanted chars to perform sentiment
CLEAN_CHARS = " abcdefghijklmnopqrstuvwxyzàáèéìíòóùúçñABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÈÉÌÍÒÓÙÚÇÑ-"

#Print header
HEAD = "[ MassSentimentGatherer ] "
WHEAD = "<WARNING> "+HEAD

#Size of the objects that process environment shall generate
DATA_PART_SIZE = 2097152

#Function Executor Max Size of Parameters
RECOMENDED_LEN_FDATA = 3670016
MAX_LEN_FDATA = 4194304
