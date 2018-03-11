import nltk

#load data
filename = "dickens_94_tale_of_two_cities.txt"

file = open(filename, 'rt')
text = file.read()
file.close()

#split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)

out_file = "tokenized_" + filename 
file_handler = open(out_file,"w")

for item in tokens:
  file_handler.write("%s\n" % item)

file_handler.close()


#print(tokens)
