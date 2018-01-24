from numpy import array
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

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

# load doc, clean and return [a single] line of tokens [for the doc] -- 
# why line of tokens? suspect each line would be vectorized
def doc_to_line(filename, vocab):
	# load the doc
	doc = load_doc(filename)
	# clean doc
	tokens = clean_doc(doc)
	# filter by vocab
	tokens = [w for w in tokens if w in vocab]
	return ' '.join(tokens) # return string of white space separated tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
	lines = list()
	# walk through all files in the folder
	for filename in listdir(directory):
		# skip any reviews in the test set
		if is_trian and filename.startswith('cv9'):
			continue
		if not is_trian and not filename.startswith('cv9'):
			continue
		# create the full path of the file to open
		path = directory + '/' + filename
		# load and clean the doc
		line = doc_to_line(path, vocab)
		# add to list
		lines.append(line) # each line represents a list of tokens for a doc
	return lines #what is the data type of lines? is it a list of things?

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab) #creating a set data type

# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, True) #vocab is being used to filter
negative_lines = process_docs('txt_sentoken/neg', vocab, True)
# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
docs = negative_lines + positive_lines
#size of docs = 900 docs + 900 docs = 1800 docs

tokenizer.fit_on_texts(docs)
# encode training data set
Xtrain = tokenizer.texts_to_matrix(docs, mode='freq') # convert the sentences directly to equal size array
# shape of Xtrain: 1800 rows x 25,768 columns

# first 900 docs have negative or 0 value, while the remaining 900 docs have positive for value
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)]) #what is this operation?
# shape of ytrain: 1800 rows x 1 column
print("Xtrain.shape,ytrain.shape:", Xtrain.shape, ytrain.shape)

# load all test reviews
positive_lines = process_docs('txt_sentoken/pos', vocab, False)
negative_lines = process_docs('txt_sentoken/neg', vocab, False)
docs = negative_lines + positive_lines
# encode training data set
Xtest = tokenizer.texts_to_matrix(docs, mode='freq')
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

print("Xtest.shape,ytest.shape:", Xtest.shape, ytest.shape)

n_words = Xtest.shape[1] # +1 for bias
# define network
model = Sequential() # why is it called Sequential? feed forward nn?
model.add(Dense(50, input_dim=n_words, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, epochs=50, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))