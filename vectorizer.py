from gensim import corpora
from nltk import word_tokenize
from tqdm import tqdm
from csv import DictReader
from collections import defaultdict



def get_most_common_words():
	most_common_words = set()
	with open("most_common_words.txt") as infile:
		for line in infile:
			most_common_words.add(line.strip("\n").lower())
	return most_common_words

def doc_generator():
	most_common_words = get_most_common_words()
	with open("tokenized_messages.csv") as infile:
		texts = []
		reader = DictReader(infile)
		count = 0
		for line in tqdm(reader):
			count +=1 
			tokens = line["Tokens"].lower().split()
			tokens = [token for token in tokens if token not in most_common_words and token.isalpha()]
			yield tokens 

def corpus_generator():
    for doc in doc_generator():
        yield dictionary.doc2bow(doc)


if __name__=="__main__":
	dictionary = corpora.Dictionary(doc_generator())
	once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 1000]
	dictionary.filter_tokens(once_ids) 
	dictionary.compactify()
	print(dictionary)
	dictionary.save(open('tokens.dict', "wb"))
	corpora.MmCorpus.serialize('messages.mm', corpus_generator()) 
