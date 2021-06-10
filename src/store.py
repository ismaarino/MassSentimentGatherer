import globals as g
import time, threading
from lithops import Storage

class CloudDataPublisher():

	def __init__(self, pf):
		self.run = False
		self.wait_time = 0
		self.pubfunc = pf

	def start(self):
		self.pending = []
		self.run = True
		while self.run:
			time.sleep(g.PUB_INTERVAL)
			self.wait_time += g.PUB_INTERVAL
			while len(self.pending) > 0 and self.run:
				self.pubfunc(self.pending.pop(0))
				self.wait_time = 0

	def stop(self):
		self.run = False


	def add(self, data):
		if self.run:
			self.pending.append(data)

	def add_array(self, data):
		if self.run:
			self.pending + data


class CloudCSVPublisher(CloudDataPublisher) :

	def __init__(self):
		super().__init__(self.publish_one)

	def publish_one(self, elem):
		print("CSV STORE: "+str(elem))
		#store elem a IBM Cloud


class CloudObjectPublisher(CloudDataPublisher) :

	def __init__(self, bucket_name):
		self.storage = Storage()
		self.bucket_name = bucket_name
		self.obj_id = self.storage.put_cloudobject("", self.bucket_name, g.CLOUD_OBJ_NAME)
		self.buffer = []
		super().__init__(self.publish_one)

	def publish_one(self, elem):
		self.buffer.append(elem)
		if len(self.buffer) > g.CLOUD_OBJ_PUB_BUFFER_SIZE or super().wait_time > g.PUB_INTERVAL*20 :
			existing = self.storage.get_cloudobject(self.obj_id).decode()
			updated = existing
			for e in self.buffer:
				updated += e+"\n"
			self.obj_id = self.storage.put_cloudobject(updated, self.bucket_name, g.CLOUD_OBJ_NAME)
			self.buffer.clear()
