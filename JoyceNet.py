import numpy
import sys
import random
import os.path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

text = open('finneganswake.txt')
data = text.read()
chars = list(set(data))

output_length = len(chars)
dataset_length = len(data)

character_map = {char : pos for pos, char in enumerate(chars)}
inverse_character_map = {pos : char for pos, char in enumerate(chars)}

sequence_length = 50

#Convert a character into a one-hot encoded vector.
def charToVector(input_char, character_map):
	Y = numpy.zeros(output_length,dtype=numpy.int8)
	Y[character_map[input_char]] = 1
	return Y

#Decode a one-hot encoded vector into a character.
def vectorToChar(input_vector, inverse_character_map):
	char = inverse_character_map[numpy.where(input_vector == 1)[0][0]]
	return char

#Generative output for the model.
def hallucinate(model, sample_string, output_string_length):
	if (len(sample_string) != sequence_length):
		print "Input string must be of length {}".format(sequence_length)
		return

	hallucination = sample_string
	char_matrix = numpy.empty((1, sequence_length, output_length), dtype=numpy.int8)
	#char = random.choice(chars)

	for i in range(output_string_length):
		for j in range(sequence_length):
			char_vector = charToVector(sample_string[j],character_map)
			char_matrix[0,j,:] = char_vector.flatten()

		#char_vector = charToVector(char, character_map)
		#char_vector = numpy.reshape(char_vector, (1,1,output_length))

		prediction = model.predict(char_matrix)

		index = numpy.argmax(prediction)
		char_vector = numpy.zeros(output_length,dtype=numpy.int8)
		char_vector[index] = 1
		char = vectorToChar(char_vector, inverse_character_map)

		hallucination = hallucination + char
		sample_string = hallucination[i:]

	return hallucination

#Creating a model based EXCLUSIVELY on lines from finnegan's wake.
def createModel():
	#Create the input matrix for the LSTM
	#Converts all the characters in the input sequence to one-hot encoded vectors
	#for the input data and output data.
	
	dataX = numpy.empty(
		(output_length,dataset_length-sequence_length, sequence_length),
	 	dtype=numpy.int8)
	
	for i in xrange(dataset_length - sequence_length):
		for j in xrange(sequence_length):
			dataX[:,i,j] = charToVector(data[i+j], character_map).flatten()

	dataY = numpy.empty((output_length,dataset_length-sequence_length), dtype=numpy.int8)
	
	for i in xrange(dataset_length-sequence_length):
		dataY[:,i] = charToVector(data[i+sequence_length], character_map).flatten()

	#Reshape the data for processing by the LSTM NN
	
	dataX = numpy.reshape(dataX, (dataX.shape[1], dataX.shape[2], dataX.shape[0]))
	dataY = numpy.reshape(dataY, (dataY.shape[1], dataY.shape[0]))

	# =========== Model Architecture ===========
	
	model = Sequential()	
	model.add(LSTM(256,input_shape = (dataX.shape[1], dataX.shape[2]),
					return_sequences = True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(dataY.shape[1], activation = 'softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam')
	model.fit(dataX, dataY, nb_epoch=50, batch_size=128, verbose = 1)

	return model
		

def main(model_name):
	#Check if the model passed in via command line already exists in
	#the current directory. Create one and save it if it isn't, otherwise
	#just load it.

	if(not os.path.isfile(model_name)):
		print "\n Model name not found. Creating new model {}\n".format(model_name)
		model = createModel()
		model.save(model_name)
	else:
		print "\n Model name found. Loading model {}\n".format(model_name)
		model = load_model(model_name)
		
	output_string_length = input("Input the desired length of output string: ")
	seed = input("Enter a number from 0 to {}: ".format(dataset_length-sequence_length-1))

	print hallucinate(model, data[seed:seed+sequence_length], output_string_length)

if __name__ == '__main__':
	main(sys.argv[1])