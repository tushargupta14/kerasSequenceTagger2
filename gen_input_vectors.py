## Script to convert the sentences into embedding vectors for input into the model 
## Words not found in 

import json 
from load_data import * 


def load_data(path_to_data = "data/") :

	with open(path_to_data + "sentence_X_dict.json","rb+") as f : 
		sentence_X_dict = json.load(f)

	with open(path_to_data + "corpus_vocab.json","rb+") as f:
		vocab = json.load(f)

	return sentence_X_dict,vocab


def get_embedding(tokens,corpus_vocab_dict): 

	sentence_embedding = []
	for token in tokens :

		if token in corpus_vocab_dict :
			sentence_embedding.append(corpus_vocab_dict[token])

	return sentence_embedding

	#return [corpus_vocab_dict[token] for token in tokens if token in corpus_vocab_dict]

def embed_sentences(sentence_X_dict,corpus_vocab_dict) :

	embedding_dict = {}

	for k,v in sentence_X_dict.iteritems() :

		embedding_dict[k] = get_embedding(v[0],corpus_vocab_dict)
	#embedding_dict = {k:get_embedding(v[0],corpus_vocab_dict) for k,v in sentence_X_dict.iteritems()}

	print "Saving the dictionary..."
	with open(path_to_data + "embedding_dict.json","wb+") as f :

		json.dump(embedding_dict,f)

if __name__ == "__main__" :

	sentence_X_dict = load_sentence_X_dict()

	corpus_vocab_dict = load_corpus_vocab()

	embed_sentences(sentence_X_dict,corpus_vocab_dict)
