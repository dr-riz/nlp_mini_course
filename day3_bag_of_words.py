import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer

# define 5 documents

docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!']

############################
# preprocessing
############################

stop_words = stopwords.words('english')
porter = PorterStemmer()

docs_preproc=[]

for item in docs:
  tokens = word_tokenize(item)
  words = [word for word in tokens if word.isalpha()]
  words = [w for w in words if not w in stop_words]
  words = [porter.stem(word) for word in words]
  doc = (" ".join(w for w in words))
  print(doc)   
  docs_preproc.append(doc)
  
  
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs_preproc)

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)
