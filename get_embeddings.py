## First 
## Script to get the vectors, embedding matrix and vocabulary  from the glove vectors 
## Give path to the vocab and vectors file

import numpy as np 
import cPickle as cp 
import json
from collections import defaultdict

def load_files(path_to_files = ""):

	print "Opening vocab file"
	
	with open(path_to_files + "vocab.txt","rb+") as f:
		words = [x.rstrip().split(' ')[0] for x in f.readlines()]

	count = 0

	vectors = defaultdict(list)
	with open(path_to_files + 'vectors.txt',"rb+") as f:
		for line in f:
			count+=	1
			vals = line.rstrip().split(' ')

			vectors[vals[0]] = [float(x) for x in vals[1:]]

			print count

	print "Loading vector dict..."
	"""with open("data/vector_dict.json","rb+") as f:
		vectors = json.load(f)"""

	print "Vocab length:",len(words),len(vectors)
	


	return words,vectors



def retrieve_embeddings(words,vectors) :

	## the vector at the idx 0 of the embedding matrix is [0 0 0 0 0 ..]
	vocab = {w: idx for idx, w in enumerate(words)}

	ivocab = {idx : w for idx, w in enumerate(words)}

	print "vocab made"
	with open("data/universal_vocab.json","wb+") as f:
		json.dump(vocab,f)

	with open("data/universal_inverse_vocab.json","wb+") as f:
		json.dump(ivocab,f)

	print "vocab saved"

	vector_dim  = len(vectors[ivocab[1]])


	W = np.zeros((len(vocab),vector_dim))

	

	print "Creating the embedding matrix"
	count = 0 
	for word,idx in vocab.iteritems() :
		count+=1
		try :
			W[idx,:] = np.asarray(vectors[word])
		except Exception as ex:

			print ex
		print count,len(vectors)

	print "Created the embedding matrix"

	W_norm = np.zeros(W.shape)

	d = (np.sum(W **2,1) **(0.5))

	W_norm  = (W.T / d).T
	print W_norm[0,:]
	return W_norm

if __name__ == "__main__" :

	words,vectors = load_files("/data2/research_projects/glove_master_for_parser/GloVe/")
	
	with open("data/vector_dict.json","wb+") as f:
		json.dump(vectors,f)
		
	embedding_matrix = retrieve_embeddings(words,vectors)

	## saving the embedding matrix here
	print "Saving files..." 
	with open("data/embedding_matrix.pkl","wb+") as f:

		cp.dump(embedding_matrix,f)
	

	print "Done"
	
