# All library imports
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Input, Dropout, Bidirectional, GRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM, SpatialDropout1D
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import os

#----------------------------------------------------------------------------------------------------------------------
def model_picker(choice, num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH):
    """
    This function acts as a model picker so that we dont need to run different files for each model. We can just make
    a choice in this, or run the models on a loop so that it can be left to run overnight or something. Also, it is
    easy to add more models to this list.
    :param choice: which model (choice) is being picked to be executed
    :param num_words: total number of words to be used in the embedding
    :param EMBEDDING_DIM: dimension of the embedding
    :param embedding_matrix: pretrained embedding matrix, in this case, GloVe
    :param MAX_SEQUENCE_LENGTH: maximum sequence length to be trained
    :return: the model which has been trained
    """

    model = Sequential()
    model.add(Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, \
                        trainable=False))
    model.add(SpatialDropout1D(0.3))

    if choice == 0:
        # This option is for a convolutional + LSTM model
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        filepath = 'checkpoints/weights.{epoch:02d}-{val_loss:.2f}_conv_lstm.hdf5'

    elif choice == 1:
        # This is for just a LSTM model
        model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        filepath = 'checkpoints/weights.{epoch:02d}-{val_loss:.2f}_just_lstm.hdf5'

    elif choice == 2:
        # This for a BiLSTM model (instead of the LSTM in previous case, use a BiLSTM
        model.add(Bidirectional(LSTM(100, dropout=0.3, recurrent_dropout=0.3)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        filepath = 'checkpoints/weights.{epoch:02d}-{val_loss:.2f}_just_bilstm.hdf5'

    elif choice == 3:
        # This is for a GRU model instead
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.8))
        model.add(Dense(3, activation='softmax'))
        filepath = 'checkpoints/weights.{epoch:02d}-{val_loss:.2f}_just_gru.hdf5'

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, filepath

#----------------------------------------------------------------------------------------------------------------------
# Loading the data
train_csv = pd.read_csv('../data/train.csv')
test_csv = pd.read_csv('../data/test.csv')
print (train_csv.shape, test_csv.shape)

# Let us display the contents a bit
# print (train_csv.head(5))
# print (test_csv.head(5))

# Setting up the labels as in the example
labels_string = train_csv['author']
# print (labels_string.shape, labels_string)

# Getting the numerical labels from the string ones
le = preprocessing.LabelEncoder()
le.fit(labels_string)
labels_num = le.transform(labels_string)
# print (labels_num)

# Converting it to categorical as needed by Keras
labels = np_utils.to_categorical(np.asarray(labels_num))
# print (labels, labels.shape)

#----------------------------------------------------------------------------------------------------------------------
# Preparing for computing the Glove embedding (as per the tutorial on Keras website)
# Downloaded and used the 100d Glove embedding. There are other choices of course - like 50,200,300.
# Used 100d since I don't have access to a GPU

train_text = []  # list of text samples
test_text = []

for i in train_csv['text']:
    train_text.append(i)

for i in test_csv['text']:
    test_text.append(i)

print (type(train_text), type(test_text))
print (type(train_text + test_text))

MAX_NUM_WORDS = 20000
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_text + test_text) 
train_sequences = tokenizer.texts_to_sequences(train_text)
print (train_sequences)
exit(0)

word_index = tokenizer.word_index
print('Found %s unique tokens in train set' % len(word_index))

# Doing the same for test set as well; using the same numbering system
test_sequences = tokenizer.texts_to_sequences(test_text)

MAX_SEQUENCE_LENGTH = 100 # 861
train_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', train_data.shape, test_data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.2, random_state=42)
print ('Train and validation shapes - ', X_train.shape, X_val.shape, y_train.shape, y_val.shape)

#----------------------------------------------------------------------------------------------------------------------
# Preparing the embeddings
embeddings_index = {}
GLOVE_DIR = '../data/'
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
EMBEDDING_DIM = 100
num_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#----------------------------------------------------------------------------------------------------------------------
print('Training models')
for i in range (0, 4):
    print ('Choice picked this time - ', i)
    model, filepath = model_picker(i, num_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH)

    # Callbacks
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, \
                                       save_weights_only=False, mode='auto', period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, \
                                  mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    print ('Fitting the model')
    model.fit(X_train, y_train, batch_size=256, epochs=50, validation_data=(X_val, y_val), verbose=1, \
              callbacks=[model_checkpoint, reduce_lr])
    
print ('All jobs complete')