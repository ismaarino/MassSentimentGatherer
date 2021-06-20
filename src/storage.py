######################################################################
#                  Pràctica Sistemes Distribuïts URV                 #
######################################################################
#                         Fitxer: storage.py                         #
#                         Autor: Ismael Curto                        #
######################################################################
import globals as g
import time, uuid, copy, json, itertools, h5py, io
import pandas as pd
from datetime import datetime, date
import tools
from lithops.storage.cloud_proxy import CloudStorage, CloudFileProxy

class CloudPublisher:

	def __init__(self, name, extension, lithops_conf):
		self.filekey = name+"_"+str(datetime.now().strftime("%b-%d-%Y"))+"____"+str(uuid.uuid1())+"."+str(extension)
		self.cloud_file_proxy = CloudFileProxy(CloudStorage(lithops_conf))
		self.file = self.cloud_file_proxy.open(self.filekey, 'w')
		self.first = True

	def close(self):
		self.file.close()

	def commit(self, data):
		pass

	def key(self):
		return self.filekey

class CloudDataFramePublisher(CloudPublisher):

	def commit(self, data):
		if self.first :
			data.to_csv(self.file, encoding='utf-8', header=True )
			self.first = False
		else :
			data.to_csv(self.file, encoding='utf-8', header=False )

class CloudRawDataPublisher(CloudPublisher):

	def commit(self, data):
		if self.first :
			self.file.write(data)
			self.first = False
		else :
			self.file.write("\n")
			self.file.write(data)


class CloudReader:

	def __init__(self, object_name, lithops_conf, mode):
		self.object_name = object_name
		self.cloud_file_proxy = CloudFileProxy(CloudStorage(lithops_conf))
		self.file = self.cloud_file_proxy.open(self.object_name, mode)

	def file(self):
		return self.file

	def close(self):
		self.file.close()

	def len(self):
		for i, l in enumerate(self.file):
			pass
		return i + 1

	def getH5(self):
		pass

	def readlines(self):
		pass

	def readlines(self, initindex, length):
		pass

	def readline(self, index):
		pass

	def dataframe(self, initindex, length):
		pass

class CloudRAWReader(CloudReader):

	def readlines(self):
		return self.file.readlines()

	def readlines(self, initindex, length):
		data = []
		for line in itertools.islice(self.file, initindex, length):
			data.append(line)
		return data

	def readline(self):
		return self.file.readline()


class CloudDFReader(CloudReader):

	def dataframe(self, initindex, length):
		data = tools.normalize(self.file.readline().decode("utf-8","replace").encode("utf-8","replace").decode("utf-8","replace"))
		print(data)
		for line in itertools.islice(self.file, initindex, length):
			data = data + "\n" + tools.normalize(line.decode("utf-8","replace").encode("utf-8","replace").decode("utf-8","replace"))
		bdata = io.StringIO(data)
		return pd.read_csv(bdata, sep=",")


class CloudH5Reader(CloudReader):

	def getH5(self):
		return h5py.File(self.file, 'r')
