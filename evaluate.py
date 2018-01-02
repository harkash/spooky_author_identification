"""
This is to evaluate the performance of the trained model. Predicts the labels for the test set
"""

# All library imports
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os

# Loading the test data and doing the pre-processing
train_csv = pd.read_csv('../data/train.csv')
test_csv = pd.read_csv('../data/test.csv')

train_text = []
for i in train_csv['text']:
    train_text.append(i)

test_text = []
for i in test_csv['text']:
    test_text.append(i)

print('Found %s texts. ', len(test_text), len(train_text))

MAX_NUM_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_text + test_text)
train_sequences = tokenizer.texts_to_sequences(train_text)

word_index = tokenizer.word_index
print('Found %s unique tokens in train set' % len(word_index))

# Doing the same for test set as well; using the same numbering system
test_sequences = tokenizer.texts_to_sequences(test_text)

MAX_SEQUENCE_LENGTH = 100 # This is what I observed from the train and test data. The max is 861 in train.
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', train_data.shape, test_data.shape)

# Loading the pretrained model and obtaining the predictions
print ('loading the saved model -->')

# We load the checkpoint with the least training loss. This is an example of that.
model = load_model('checkpoints/weights.48-0.58_just_lstm.hdf5')

print ('Doing the prediction -->')
preds = model.predict(test_data)
print (preds.shape)

# Lets first see what the ground truth mappings for labels are
le = preprocessing.LabelEncoder()
le.fit(train_csv['author'])
print (list(le.inverse_transform([0,1,2]))) # EAP, HPL<, MWS

test_csv['EAP'] = preds[:,0]
test_csv['HPL'] = preds[:,1]
test_csv['MWS'] = preds[:,2]

# Turns out the submission file wants the id to be within double quotes. 
new_id = []
for i in test_csv['id']:
    to_append = "\"" + i + "\""
    new_id.append(to_append)
    
test_csv['new_id'] = new_id
print (test_csv.head(5))

to_write = ['new_id', 'EAP', 'HPL', 'MWS']

# Saving it into a csv file. The name is kept consistent with the model used to evaluate
test_csv.to_csv('test_predictions_just_lstm.csv', columns=to_write, index=False)

