import globals as g
import time, uuid, copy, json
from datetime import datetime, date
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

	def __init__(self, object_name, lithops_conf):
		self.object_name = object_name
		self.cloud_file_proxy = CloudFileProxy(CloudStorage(lithops_conf))
		self.file = self.cloud_file_proxy.open(self.object_name, 'r')

	def close(self):
		self.file.close()

	def readlines(self):
		return self.file.readlines()
