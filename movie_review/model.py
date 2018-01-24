from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json
import pickle

# load doc into memory -- as is with new lines etc.
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', punctuation) # this sets up the punctuation replacement
	tokens = [w.translate(table) for w in tokens] #this removes the punctuation
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# filter out stop words
	stop_words = set(stopwords.words('english'))
	tokens = [w for w in tokens if not w in stop_words]
	# filter out short tokens
	tokens = [word for word in tokens if len(word) > 1]
	return tokens #returns a list of tokens in 1 line
    
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tkizer = pickle.load(handle)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

#lss, accuracy = loaded_model.evaluate(Xtest, ytest, verbose=0)
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print('Test Accuracy: %f' % (accuracy*100))

# classify a review as negative (0) or positive (1)
def predict_sentiment(review, vocab, tokenizer, model):
	# clean
	tokens = clean_doc(review)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	# convert to line
	line = ' '.join(tokens)
	# encode
	encoded = tokenizer.texts_to_matrix([line], mode='freq')
	# prediction
	yhat = loaded_model.predict(encoded, verbose=0)
	return round(yhat[0,0])

# load the vocabulary
vocab_filename = 'vocab.txt'
voc = load_doc(vocab_filename)
voc = voc.split()
voc = set(voc) #creating a set data type
	
# test positive text
text = 'Best movie ever!'
print(predict_sentiment(text, voc, tkizer, loaded_model))
# test negative text
text = 'This is a bad movie.'
print(predict_sentiment(text, voc, tkizer, loaded_model))


