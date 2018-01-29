# Creates corpus for input to fastText for unsupervised learning of word embeddings, mergeing all the jds

import sys
import json
import os 
import re 
def create_copus(sentence_X_dict) :


	corpus_file = open("data/sentence_corpus.txt","a")
	for k,v in sentence_X_dict.iteritems():
		corpus_file.write(v[1]+"\n")


def file_merger(path_to_parent_folder = "") :


	for dirname,dirs,files in os.walk(path_to_parent_folder) :
		for filename in files :

			if re.search(filename , '')




if __name__ == "__main__" :

	create_copus(json.load(open("data/sentence_X_dict.json")))