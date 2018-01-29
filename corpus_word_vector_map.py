## Creates mapping for words of the corpus from the universal vocabulary 
## manually add 'PAD' and 'UNK' to corpus_word_vocab and corpus_vector_dict 

## Only those words are stored in corpus_word_vocab which are present in the vectors
from load_data import * 


def create_map():

	vectors = load_vector_dict()

	vocab = load_universal_vocab()

	corpus_vector_dict = {}

	corpus_word_vocab = load_corpus_vocab()
	##intialise unk_vector and pad_vec
	unk_vec = vectors["unk"]
	pad_vec = vectors["ukn"]
	count = 0 

	for word,w_id in corpus_word_vocab.iteritems() :

		if word in vectors :
			corpus_vector_dict[w_id] = vectors[word]

		else : 
			corpus_vector_dict[w_id] = unk_vec
			count+=1
		print count,len(corpus_word_vocab)
	corpus_vector_dict[0] = pad_vec
	with open("data/corpus_vector_dict.json","wb+") as f:

		json.dump(corpus_vector_dict,f)


if __name__ == "__main__" :

	create_map()