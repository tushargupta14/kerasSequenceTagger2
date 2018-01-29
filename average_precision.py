## Script for loading a model and evaluation 
## Loads Model for epoch 4 for Model 2 pre-trained embeddings with trainable = True
## Change the model number for calculating average Precision 
## Change the folder to the weights file
import keras 
import h5py
import json 
import cPickle as cp 
import numpy as np 

from keras.models import model_from_json
from sklearn import metrics 


def load_model(path_to_model = "weights_train_false/") :

	with open(path_to_model + "model2_epoch_4.json","rb+") as f:
		loaded_model_json  = f.read()

	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights(path_to_model + "model2_weights_epoch_4.h5")

	with open("data/test_batch_dict_new.json","rb+") as f :

		test_batch_dict_new = json.load(f)


	with open("data/embedding_dict.json","rb+") as f :

		embedding_dict = json.load(f)
	with open("data/sentence_Y_dict.json","rb+") as f :

		sentence_Y_dict = json.load(f)
	return loaded_model,test_batch_dict_new,embedding_dict,sentence_Y_dict

def pad_zeros(ids,embedding_dict) :
	## return a numpy array of all the sentences in the batch padded with zeros for their embeddings 

	max_len = 0 
	for sent_id in ids :
		embedding = embedding_dict[sent_id]
		if len(embedding) > max_len :
			max_len = len(embedding)

	#print "max_len", max_len


	sent_mat = [embedding_dict[sent_id] for sent_id in ids]

	if len(np.array(sent_mat).shape) == 1 :
		return np.array([xi + [0]*(max_len - len(xi)) for xi in sent_mat]),max_len

	else :

		padded_mat = np.zeros((len(ids),max_len))

		sent_mat = np.array(sent_mat)
		for i in xrange(len(ids)) :
	 		padded_mat[i,:] = np.pad(sent_mat[i,:],(0,max_len - sent_mat[i,:].shape[0]),'constant',constant_values = (0))


		return padded_mat,max_len


def convert_to_onehot(batch_labels) :

	output_mat = []
	for sent_labels in batch_labels :
		temp_mat = []	
		for word_label in sent_labels :
			if word_label == -1 :

				temp = [0]*5
				temp_mat.append(temp)
				continue
			temp = [0]*5
			temp[word_label] = 1
			temp_mat.append(temp)

		output_mat.append(temp_mat)

	return np.asarray(output_mat)

def load_utility_dicts(path_to_data  = "data/") :

	with open(path_to_data + "sentence_X_dict.json","rb+") as f : 
		sentence_X_dict = json.load(f)

	""""with open(path_to_data + "word_vocab.json","rb+") as f:
		word_vocab_dict = json.load(f)"""

	with open(path_to_data + "labels.json","rb+") as f:
		label_dict = json.load(f)

	"""with open(path_to_data+"embedding_dict.json") as f :

		embedding_dict = json.load(f)"""
	return label_dict,sentence_X_dict


def get_prediction_labels(predictions,sent_ids):


	predict_mat = []
	for i in xrange(len(sent_ids)) :
		#s_id   = sent_ids[i]

		predicted_sent = list(predictions[i])

		#prob_list  = [max(list(labels)) for labels in predicted_sent]
		predict_mat.append([list(labels).index(max(labels)) for labels in predicted_sent])


		#label_list = [ilabel_dict[int(el)] for el in label_list]
		#true_labels  = sentence_Y_dict[s_id]
	
		#true_labels = [ilabel_dict[el] for el in true_labels]
	
		#word_list = [idx for word in sents[str(i)]]
		#print "sentence : ",sentence_X_dict[s_id][0]
		#print "predicted :",label_list
		#print "true :",true_labels
		
		#for idx in range(len(sentence_X_dict[s_id][0])) :
			#print sentence_X_dict[s_id][0][idx],true_labels[idx],label_list[idx],prob_list[idx]
	return predict_mat	

def evaluation(model,test_batch_dict,embedding_dict,sentence_Y_dict) :


	model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy',metrics = ['binary_accuracy'])

	batch_count  = 0
	weighted_average_precision = 0 
	micro_average_precision = 0 
	macro_average_precision = 0 
	scores = []
	for batch_n, sent_ids in test_batch_dict.iteritems() :

		batch_count+=1 

		sents, max_len = pad_zeros(sent_ids,embedding_dict)

		true_labels = [sentence_Y_dict[item] + [-1] * (max_len - len(sentence_Y_dict[item])) for item in sent_ids] 
		#print np.array(true_labels).shape

		true_labels = np.asarray(true_labels).flatten()
	#####
		predictions = model.predict_on_batch(sents)

		#print np.array(predictions).shape

		predictions = get_prediction_labels(predictions,sent_ids)

		#print np.array(predictions).shape

		predicted_labels = np.asarray(predictions).flatten()
		weighted_average_precision += metrics.precision_score(true_labels,predicted_labels,labels = [0,1,2,3,4],average = "weighted")

		macro_average_precision += metrics.precision_score(true_labels,predicted_labels,labels = [0,1,2,3,4],average = "macro")


		if batch_count %10 == 0 :
			print "wap :", weighted_average_precision / batch_count,"MAP :",macro_average_precision / batch_count


	print "For Test Dataset .15 of Total "
	print "wap :", weighted_average_precision / batch_count,"MAP :",macro_average_precision / batch_count


if __name__ =="__main__" :

	model,test_batch_dict_new,embedding_dict,sentence_Y_dict = load_model("weights/")

	evaluation(model,test_batch_dict_new,embedding_dict,sentence_Y_dict)
