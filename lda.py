from gensim.models.ldaseqmodel import LdaSeqModel
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from nltk import word_tokenize
from tqdm import tqdm
from csv import DictReader
from collections import defaultdict
import pprint 

pp = pprint.PrettyPrinter(indent=4)

id2word = corpora.Dictionary.load('tokens.dict')
mm = corpora.MmCorpus('messages.mm')
ldaseq = LdaModel(corpus=mm, id2word= id2word, num_topics=15)
pp.pprint(ldaseq.print_topics())
ldaseq.save("lda_model")
