## Script for testing 

import json 
from collections import defaultdict 


print "Opening vocab file"
	
path_to_files = "/data2/research_projects/glove_master_for_parser/GloVe/"

with open(path_to_files + "vocab8_aug.txt","rb+") as f:
	words = [x.rstrip().split(' ')[0] for x in f.readlines()]

count = 0

vectors = defaultdict(int)
with open(path_to_files + 'vectors_8aug.txt',"rb+") as f:

	for line in f:
		count+=	1
		vals = line.rstrip().split(' ')
		vectors[vals[0]]+=1
		if count%100 ==0 :
			print count 

with open("data/vector_count_dict.json","wb+") as f:
		json.dump(vectors,f)