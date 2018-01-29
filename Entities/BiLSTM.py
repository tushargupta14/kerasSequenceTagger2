## Pre Trained BiLSTM model 


import keras 
import json 
from keras.models import Sequential 
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM 
from keras.layers.core import Dense,Dropout,Activation
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras import optimizers
import cPickle as cp 
import numpy as np 
import h5py
from load_data import *
from keras import metrics
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def convert_to_onehot(batch_labels) :


	### Label -1 assigned to the pad values and before input and mapped  to [0 0 0 0 0]
	output_mat = []
	for sent_labels in batch_labels :
		temp_mat = []	
		for word_label in sent_labels :
			if word_label == -1:

				temp = [0]*43
				temp_mat.append(temp)
				continue
			temp = [0]*43
			temp[word_label] = 1
			temp_mat.append(temp)

		output_mat.append(temp_mat)

	return np.asarray(output_mat)

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

def test_on_validation_data(model,sentence_Y_dict, validation_batch_dict_new,embedding_dict,use_whole = False) :

	batch_count = 0 
	validation_loss = 0 
	acc  = 0 
	for batch_n,sent_ids in validation_batch_dict_new.iteritems() :

		vList = []
		batch_count +=1 

		sents , max_len = pad_zeros(sent_ids,embedding_dict)

		labels = [sentence_Y_dict[item]+[-1]*(max_len - len(sentence_Y_dict[item])) for item in sent_ids] 
				
		labels = convert_to_onehot(labels)

		vList = model.test_on_batch(sents,labels)
		validation_loss+= vList[0]
		acc+=vList[1]

		if batch_count == 10 and use_whole == False :
			break 


	print "validation_loss :",validation_loss/batch_count
	acc = acc/batch_count 
	print "Accuracy :",acc
	return [validation_loss/batch_count,acc]

def create_embedding_matrix(corpus_word_vocab) :

	corpus_vector_dict = load_corpus_vector_dict()
	embed_mat = np.zeros((len(corpus_word_vocab),300))
	for k,v in corpus_word_vocab.iteritems() :

		embed_mat[v,:] = np.asarray(corpus_vector_dict[k]) 
	return embed_mat

def train_model():

	#### Define models here 
	print "Loading data..."
	embedding_dict = load_embedding_dict()
	sentence_Y_dict = load_sentence_Y_dict()
	corpus_word_vocab = load_corpus_vocab()
	print "Loading batch dicts"
	train_batch_dict_new = load_train_data()
	validation_batch_dict_new = load_validation_data()


	n_classes = 43

	input_dim = len(corpus_word_vocab) ## Highest index + 1  becuase includes 0 
	output_dim  = 300

	embed_mat = create_embedding_matrix(corpus_word_vocab)
	model = Sequential()
	model.add(Embedding(input_dim,output_dim,weights = [embed_mat],mask_zero = True))
	model.add(Dropout(0.3))
	model.add(Bidirectional(LSTM(300,activation = 'tanh',return_sequences = 'True')))

	model.add(TimeDistributed(Dense(n_classes,activation = 'softmax')))

	rmsprop = optimizers.RMSprop(lr = 0.0000001)

	model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['acc',metrics.categorical_accuracy])
	model.layers[1].trainable= False
	epochs = 10	

	avgLoss_list = []
	vLoss_list = []
	score_list = []
	
	print model.metrics_names

	for i in range(epochs) :

		print "epoch",i
		avgLoss = 0 
		batch_count = 0 
		for batch,sent_ids in train_batch_dict_new.iteritems() :

			batch_count+= 1
			#print "batch:",batch_count

			sents , max_len = pad_zeros(sent_ids,embedding_dict)

			#sents = convert_to_vectors(sents,max_len)
			labels = [sentence_Y_dict[item]+[-1]*(max_len - len(sentence_Y_dict[item])) for item in sent_ids] 
			
			labels = convert_to_onehot(labels)
			#print sents.shape,labels.shape
		
			scores = model.train_on_batch(sents,labels)

			avgLoss +=scores[0]
			
			if batch_count%10 ==0:
				print "epoch :",i,"avgLoss :",avgLoss/batch_count,"n_batches : ",batch_count
				vLoss = test_on_validation_data(model,sentence_Y_dict,validation_batch_dict_new,embedding_dict) 
				
		avgLoss = avgLoss / len(train_batch_dict_new)

		avgLoss_list.append(avgLoss)

		## Evaluating the loss on the whole validation data 
		vLoss = test_on_validation_data(model,sentence_Y_dict,validation_batch_dict_new,embedding_dict,use_whole = True)
		vLoss_list.append(vLoss)
	
		print "epoch : ",i," avgLoss : ",avgLoss,"validation_loss :",vLoss[0],"Accuracy :",vLoss[1]
		score_list.append(scores)
		## Saving the model weights and architecture for each epoch 
		json_object = model.to_json()
		with open("weights/model2_epoch_"+str(i)+".json","wb+") as json_file :
			json_file.write(json_object)
		model.save_weights('weights/model2_weights_epoch_'+str(i)+'.h5')


		with open("results/avgLoss_list.pkl","wb+") as f:
			cp.dump(avgLoss_list,f)

		with open("results/score_list.pkl","wb+") as f:
			cp.dump(score_list,f)
		with open("results/vLoss_list.pkl","wb+") as f:
			cp.dump(vLoss_list,f)
if __name__ == "__main__" :

	train_model()