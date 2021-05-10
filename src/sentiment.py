import globals as g
from textblob import TextBlob
#from googletrans import Translator

class SentimentAnalizer:
	def __init__(self):
		self.translator = None#Translator()


	def clean(data):
		r = str(data)
		r2 = ""
		for char in r:
			if char in g.CLEAN_CHARS:
				r2 += char
			else:
				r2 += " "
		return r2


	def analysis(self, data):
		c_data = SentimentAnalizer.clean(data)
		#if(translator.detect(c_data) != 'en') :
			#tb = tb.translate(to='en')
		tb = TextBlob(c_data)
		return tb.sentiment.polarity