import nltk

#load data
filename = "dickens_94_tale_of_two_cities.txt"

file = open(filename, 'rt')
text = file.read()
file.close()

num_of_examples=10

#split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
print("number of tokens=" + str(len(tokens)))
print(tokens[:num_of_examples])

#preprocessing: Filter Out Punctuation
# remove all tokens that are not alphabetic. diff bw "filter out punctuation" and "extracting isalpha()"?
words = [word for word in tokens if word.isalpha()]
print("number of words after removing punctionation=" + str(len(words)))
print(words[:num_of_examples])

# preprocessing: remove stop words e.g. 'the’, ‘is’, ‘are' from title	
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#print(stop_words)

words = [w for w in words if not w in stop_words]
print("number of words after removing stop words=" + str(len(words)))
print(words[:num_of_examples])	

# preprocessing: stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
words = [porter.stem(word) for word in words]
print("number of stemmed words=" + str(len(words)))
print(words[:num_of_examples])


out_file = "tokenized_" + filename 
file_handler = open(out_file,"w")

for item in words:
  file_handler.write("%s\n" % item)
file_handler.close()


#print(tokens)
