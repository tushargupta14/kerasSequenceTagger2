### imlements loading data functions 
import cPickle as cp  
import json 

path_to_data = "/home/tushar/pre_trained/data/"

def load_universal_vocab(path_to_data = "data/") :

	with open(path_to_data + "universal_vocab.json","rb+") as f :

		 return json.load(f)
def load_vector_dict(path_to_data = "data/") :

	with open(path_to_data + "vector_dict.json","rb+") as f :

		return json.load(f)

def load_train_data(path_to_data = "data/") :

	with open(path_to_data + "train_batch_dict_new.json","rb+") as f :

		return json.load(f)

def load_validation_data(path_to_data = "data/") :

	with open(path_to_data + "validation_batch_dict_new.json","rb+") as f :

		return json.load(f)

def load_sentence_Y_dict(path_to_data = "data/") :

	with open(path_to_data + "sentence_Y_dict.json","rb+") as f :

		return json.load(f)
def load_embedding_dict(path_to_data = "data/") :

	with open(path_to_data + "embedding_dict.json") as f :
		return json.load(f)


def load_corpus_vocab(path_to_data = "data/") :

	with open(path_to_data + "corpus_word_vocab.json") as f :
		return json.load(f)

def load_corpus_vector_dict(path_to_data = "data/") :

	with open(path_to_data + "corpus_vector_dict.json") as f:
		return json.load(f)

def load_sentence_X_dict(path_to_data = "data/") :

	with open(path_to_data + "sentence_X_dict.json","rb+") as f :

		return json.load(f)