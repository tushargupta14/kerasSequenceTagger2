## word_vector map
from load_data import * 


def create_queries() :

	vocab = load_corpus_vocab()
	f = open("data/queries.txt","wb+") 
	for k,v in vocab.iteritems() :
		f.write(k+'\n')

def create_map():


	corpus_vector_dict = {}

	corpus_word_vocab = load_corpus_vocab()
	##intialise unk_vector and pad_vec

	with open("data/vectors.txt","rb+") as f :
		for line in f :
			data = line.rstrip().split()
			word = data[0]
			vector = [float(x) for x in data[1:]]

			corpus_vector_dict[word] = vector

	with open("data/corpus_vector_dict.json","wb+") as f:

		json.dump(corpus_vector_dict,f)


if __name__ == "__main__" :

	#create_queries()
	create_map()