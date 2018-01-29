## preparing intial data from raw-text

from collections import defaultdict
import os
import json 
def clean(text):

    cleaned_text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    cleaned_text = ''.join([i.rstrip() if i.isalnum() or i == "," or i == "."  else " " for i in cleaned_text])
    cleaned_text = " ".join(str(x).lower() for x in cleaned_text.split())
    cleaned_text = cleaned_text.replace(","," ,")
    cleaned_text = cleaned_text.replace(".",". ")
    return cleaned_text


def process_file(file_path,ann_path,tag_set,jd_label_map_dict) :

	print file_path,ann_path
	word_tag_dict = defaultdict(lambda : defaultdict(list))
	
	
	with open(ann_path,"rb+") as f :
		time_idx = 0 
		for line in f :
			elements = line.rstrip().split() 
			if "#" in elements[0] :
				continue
			if "T" in elements[0] :
				## we have a entity here
				data = line.rstrip().split("\t",4)
				tag = data[1].split()[0]
				tag_set.add(tag)
				text = clean(data[2])
				word_list = text.split() 
				#print text
				#print word_list
				for i in range(len(word_list)) :
					#print word_list[i]
					if i == 0 :
						word_tag_dict[time_idx][word_list[i]].append('B-'+tag)
					else :
						word_tag_dict[time_idx][word_list[i]].append('I-'+tag)
				#print word_tag_dict[time_idx]
				time_idx+=1 	
	jd_text = ""
	jd_word_list = []
	jd_label_list = []
	with open(file_path,"rb+") as f :
		for line in f :
			jd_text += clean(line)
			jd_text+= " "
	#print jd_text


	untagged_words = jd_text.split()

	t = 0 
	tag_offset = 0
	for i in range(len(jd_text.split())) :

		w = untagged_words[i]
		if tag_offset >= len(word_tag_dict[t]) :
			t+=1
			tag_offset = 0 
		#print word_tag_dict[t]
		if w in word_tag_dict[t]:
			#print "Here"  
			#print word_tag_dict[t][w]
			jd_word_list.append(w)
			jd_label_list.append(word_tag_dict[t][w][0])
			tag_offset+=1
		else :
			jd_word_list.append(w)
			jd_label_list.append('O')

	#print jd_word_list
	#print jd_label_list
	#print word_tag_dict

	jd_label_map_dict[len(jd_label_map_dict)] = [jd_word_list,jd_label_list]

def read_txt_files() :	
	tag_set = set()

	jd_label_map_dict = {}
	for dirs,subdirs,files in os.walk("/home/edge/Practice/edge/") :

		for f in files :
			if f.endswith(".txt") : 

				file_path =  os.path.join(dirs,f)
				ann_path = file_path[:-4] + ".ann"
				print file_path
				process_file(file_path,ann_path,tag_set,jd_label_map_dict)	
	return jd_label_map_dict,tag_set

def create_labels(tag_set) :

	label_dict = {}
	idx = 0
	for tag in tag_set :
		label_dict['B-'+tag] = idx 
		idx+=1 
		label_dict['I-'+tag] = idx
		idx+=1
	label_dict['O'] = idx

	print len(tag_set),len(label_dict)

	with open("labels.json","wb+") as f :
		json.dump(label_dict,f)
if __name__ == "__main__" :

	jd_label_map_dict,tag_set = read_txt_files()
	with open("data/jd_label_map_dict.json","wb+") as f :
		json.dump(jd_label_map_dict,f)
	create_labels(tag_set)