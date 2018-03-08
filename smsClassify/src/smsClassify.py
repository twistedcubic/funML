#get those libraries
import numpy as np
import pandas as pd

data = pd.read_csv("../data/sms.csv",encoding='latin-1')

texts = data['v2']

#Create data
all_labels = [1 if x == 'spam' else 0 for x in data['v1']]

num_test_samples = 50
labels = all_labels[:-num_test_samples]
test_labels = all_labels[-num_test_samples:]

text_data = []
#dictionary of word and its index 
vocab_dict = {}
#
count_dict = {}
counter = 0

#gather vocab
for text in texts:
    text_list = []
    for word in text.split(' '):
        if word in vocab_dict:
            text_list.append(vocab_dict[word])
            count_dict[word] = count_dict[word] + 1
        else:
            vocab_dict[word] = counter
            count_dict[word] = 1
            text_list.append(counter)
            counter += 1
    text_data.append(text_list)

#vocab_len for this dataset is 15586
vocab_len = len(vocab_dict)
text_data_ar = np.zeros((len(text_data),vocab_len))
counter = -1
for word_indices in text_data:
    counter += 1
    for word_index in word_indices:
        text_data_ar[counter, word_index] = 1

#test data
test_data_ar = text_data_ar[-num_test_samples:]
#train data
train_data_ar = text_data_ar[:-num_test_samples]

## Create and train model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def create_model():
    model = Sequential()
    #linear layer first
    model.add(Dense(512,input_shape=(np.shape(train_data_ar)[1],)))
    model.add(Dense(256, activation='tanh'))
    #Use sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model

model = create_model()
#Train model
model.fit(train_data_ar, labels, batch_size=len(test_data_ar), epochs=1,  verbose=1)

#predicted is an array of form: [array([[ 0.00392788]], dtype=float32), array([[ 0.07294346]], dtype=float32),...]
predicted = [model.predict(np.reshape(x,(1, vocab_len))) for x in test_data_ar]

#represent data in terms of spam or ham
predicted = [x >= 0.5 for x in predicted]
actual = [x == 1 for x in test_labels]
compare = [predicted[i]==actual[i] for i in range(len(predicted))]

print('Correct count: ', compare.count(True), ' out of total ', num_test_samples)
#After just 1 epoch, Correct count: 49 out of total 50


### Some improvements
#take the top 1500 most frequent words instead of 15586. 
#This makes training faster, reduces memory use, and makes the system more 
#scalable to bigger datasets

words_to_keep = 1500
vocab = list(vocab_dict.keys())
vocab.sort(key = lambda x : vocab_dict[x], reverse=True)
top_vocab = set(vocab[0:words_to_keep])

#gather only most frequent words
for text in texts:
    text_list = []
    for word in text.split(' '):
        if word not in top_vocab:
            continue
        if word in vocab_dict:
            text_list.append(vocab_dict[word])
        else:
            vocab_dict[word] = counter
            text_list.append(counter)
            counter += 1
    text_data.append(text_list)
    
vocab_len = len(vocab_dict)
text_data_ar = np.zeros((len(text_data),vocab_len))
counter = -1
for word_indices in text_data:
    counter += 1
    for word_index in word_indices:
        text_data_ar[counter, word_index] = 1

model_15 = create_model()
#Train model
model_15.fit(train_data_ar, labels, batch_size=len(test_data_ar), epochs=1,  verbose=1)

predicted = [model_15.predict(np.reshape(x,(1, vocab_len))) for x in test_data_ar]

#represent data in terms of spam or ham
predicted = [x >= 0.5 for x in predicted]
actual = [x == 1 for x in test_labels]
compare = [predicted[i]==actual[i] for i in range(len(predicted))]

print('Correct count: ', compare.count(True), ' out of total ', num_test_samples)
