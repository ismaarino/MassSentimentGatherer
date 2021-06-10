import globals as g
import os
import re
import random
import numpy as np
import pandas as pd
import tensorflow as tensor
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt

from numpy import array, array, asarray, zeros
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class Analizer:
	def __init__(self, lang, conf):
		self.tag_re = re.compile(r'<[^>]+>')
		self.hashtag_re = re.compile(r'#[^ ]+')
		self.mention_re = re.compile(r'@[^ ]+')
		self.url_re = re.compile(r'http[^ ]+')
		self.conf = conf
		self.model = self.set_model(lang)

	def open_files(dir, encoding, contains):
		files = []
		for filename in os.listdir(dir):
			if(contains in filename):
				files.append(open(os.path.join(dir, filename), encoding=encoding))
		return files


	def clean(self, text):
		text = bytes(text, 'utf-8').decode('utf-8','ignore')
		text = self.tag_re.sub('', text)
		text = self.mention_re.sub('', text)
		text = self.hashtag_re.sub('', text)
		text = self.url_re.sub('', text)
		text = re.sub('[^'+g.CLEAN_CHARS+']', ' ', text)
		text = re.sub(r'\s+', ' ', text).strip().lower()
		if len(text) == 0:
			raise Exception("Data becomed empty value after cleaning")
		return text

	def exist_file(dir, warn):
		try:
			f = open(dir)
			return True
		except IOError:
			if warn:
				print("\n [ERROR] Missing file "+dir)
			return False

	def generate_model(self, d_dir, model_name, tokenizer_name, lang):
		encoding = open(g.SENTIMENTS_DIR+lang+"/enc.txt").readlines()[0]
		positive_files = Analizer.open_files(g.SENTIMENTS_DIR+lang+"/", encoding, "positive")
		negative_files = Analizer.open_files(g.SENTIMENTS_DIR+lang+"/", encoding, "negative")
		neutral_files = Analizer.open_files(g.SENTIMENTS_DIR+lang+"/", encoding, "neutral")
		positive = []
		negative = []
		neutral = []
		for file in positive_files:
			positive = positive + file.readlines()
		for file in negative_files:
			negative = negative + file.readlines()
		for file in neutral_files:
			neutral = neutral + file.readlines()

		plen=len(positive)
		nlen=len(negative)
		nelen=len(neutral)

		print("\nLoaded cases {\t positive: [\n\t\t\t files: "+str(len(positive_files))+",\n\t\t\t inputs: "+str(plen)+" ]\n\t\t negative: [\n\t\t\t files: "+str(len(negative_files))+",\n\t\t\t inputs: "+str(nlen)+" ]\n\t\t neutral: [\n\t\t\t files: "+str(len(neutral_files))+",\n\t\t\t inputs: "+str(nelen)+" ]\n\t\t}")

		X = positive + negative + neutral
		y = np.concatenate((np.full(plen, 1.0), np.full(nlen, 0.0), np.full(nelen, 0.5)))

		c = list(zip(X,y))
		random.shuffle(c)
		X, y = zip(*c)
		y = np.array(y)
		print("Loaded "+str(len(y))+" Training Values")

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.conf['ml_training']['test_size'], random_state=42)

		self.model_tokenizer = Tokenizer(num_words=self.conf['ml_training']['tokenizer_words'])
		self.model_tokenizer.fit_on_texts(X_train)

		X_train = self.model_tokenizer.texts_to_sequences(X_train)
		X_test = self.model_tokenizer.texts_to_sequences(X_test)

		vocab_size = len(self.model_tokenizer.word_index)+1

		X_train = pad_sequences(X_train, padding='post', maxlen=self.conf['ml_training']['data_maxlen'])
		X_test = pad_sequences(X_test, padding='post', maxlen=self.conf['ml_training']['data_maxlen'])
		print("Tokenized Training Data")

		embeddings_dictionary = dict()
		wordv_file = open(g.WORDVECTORS_DIR+lang+"/"+lang+".vec", encoding="utf-8")
		print("Loading WordFile...")
		for line in wordv_file:
			records = line.split()
			word = records[0]
			vector_dimensions = asarray(records[1:], dtype='float32')
			embeddings_dictionary[word] = vector_dimensions
		wordv_file.close()
		print("Loaded WordFile data")

		embedding_matrix = zeros((vocab_size, self.conf['ml_training']['wordvector_dimension']))
		for word, index in self.model_tokenizer.word_index.items():
			embedding_vector = embeddings_dictionary.get(word)
			if embedding_vector is not None:
				embedding_matrix[index] = embedding_vector
		print("Embedding matrix of "+str(len(embedding_matrix))+" words")

		model = Sequential()
		adam = Adam(learning_rate=self.conf['ml_training']['learning_rate'], beta_1=0.9, beta_2=0.999, decay=0.01)
		model.add(Embedding(vocab_size, self.conf['ml_training']['wordvector_dimension'], weights=[embedding_matrix], input_length=self.data_maxlen , trainable=False))
		model.add(BatchNormalization())
		model.add(Bidirectional(LSTM(64,kernel_regularizer=keras.regularizers.l2(0.0001), dropout=0.6)))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
		print(model.summary())

		hist = model.fit(X_train, y_train, batch_size=self.conf['ml_training']['batch_size'], epochs=self.conf['ml_training']['epochs'], verbose=1, validation_split=self.conf['ml_training']['test_size'])

		os.mkdir(d_dir)
		model.save(d_dir+model_name)
		pickle.dump(self.model_tokenizer, open(d_dir+tokenizer_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

		plt.plot(hist.history['loss'])
		plt.plot(hist.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('No. epoch')
		plt.legend(['train', 'test'], loc="upper left")
		plt.show()

		plt.plot(hist.history['acc'])
		plt.plot(hist.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('No. epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.show()

		return model

	def model_summary(self):
		return self.model.summary()

	def evaluate_model(self, X_text, y_test):
		score = self.model.evaluate(X_test, y_test, verbose=1)
		print("Test Score:", score[0])
		print("Test Accuracy:", score[1])

	def set_model(self, lang):
		d_dir = g.MODELS_DIR+lang+"/"
		model_name = lang+"_model.h5"
		tokenizer_name = lang+"_tokenizer.pickle"
		if(Analizer.exist_file(d_dir+model_name, False) and Analizer.exist_file(d_dir+tokenizer_name, False)):
			self.model_tokenizer = pickle.load(open(d_dir+tokenizer_name, 'rb'))
			return keras.models.load_model(d_dir+model_name)
		elif self.check_supported(lang):
			return self.generate_model(d_dir, model_name, tokenizer_name, lang)
		else :
			raise Exception("Labeled sentiment data or Pre-trained Model must be provided to support lang: "+lang)
		return None

	def check_supported(self, lang):
		return Analizer.exist_file(g.WORDVECTORS_DIR+lang+"/"+lang+".vec",True) and Analizer.exist_file(g.SENTIMENTS_DIR+lang+"/enc.txt",True) and Analizer.exist_file(g.SENTIMENTS_DIR+lang+"/positive_"+lang+".txt",True) and Analizer.exist_file(g.SENTIMENTS_DIR+lang+"/negative_"+lang+".txt",True)

	def analysis(self, data):
		try:
			c_data = self.clean(data)
			print(c_data)
			test = self.model_tokenizer.texts_to_sequences(c_data)
			flat_list = []
			for sublist in test:
				for item in sublist:
					flat_list.append(item)
			flat_list = [flat_list]
			test = pad_sequences(flat_list, padding='post', maxlen=self.data_maxlen)
			return self.model.predict(test)[0][0]
		except Exception as e:
			return 0.50

"""
a = Analizer("es");
print(a.analysis("El covid esta peor que nunca"))
print(a.analysis("Ya se termina el covid, estoy muy contenta"))
print(a.analysis("Que asco la enfermedad nueva esta"))
print(a.analysis("Al menos con el covid nos podemos quedar en casa "))
print(a.analysis("No me gusta nada el covid"))
print(a.analysis("Gracias a las personas que salvan a otras"))
print(a.analysis("Puto covid de los cojones"))
print(a.analysis("Voy a matar a alguien como sigamos encerrados"))
print(a.analysis("Me encanta cuando me miras bb"))
print(a.analysis("Hola amigos! Hoy directito de twitch esta noche"))
print(a.analysis("Te odio"))
print(a.analysis("Estoy contento"))
print(a.analysis("Estoy triste"))
print(a.analysis("Denuncian a Vox, pero Ana Rosa y Vicente Vallés no se enterarán, sus “fuentes” solo les traen noticias sobre los cortes de pelo de Pablo Iglesias."))
print(a.analysis("La estafa que es VOX ya lo contó un Concejal del partido. "))
print(a.analysis("La noticia de que denuncian a Vox es utilizada para que se deje de hablar del escándalo de Pablo Iglesias y el CNI, o de los recortes en Sanidad para el 2024, no"))
print(a.analysis("Muere la abuela de Cristina Pedroche por covid: 'Deberían ser eternos'"))
print(a.analysis("Hoy estoy aqui para decir que me cago en todos tus muertos"))
print(a.analysis("Vale la pena vivir en este pais"))
print(a.analysis("No me gusta españa"))
print(a.analysis("Me encanta españa"))"""
