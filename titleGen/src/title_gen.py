
'''
Generate your own viral video title
'''
import numpy as np
import pandas as pd
import json

data = pd.read_csv("../data/USvideos.csv")
titles = data['title']
#total number of titles: 21965

#process data to create vectors from text
titles0 = titles

char_index_dict = {}
index_char_dict = {}
titles_data = []

tarmac_len = 15
#minimum length requirement for lstm training
min_len = tarmac_len + 1
counter = 0

for title in titles:
    if len(title) < min_len:
        continue
    title_data = []
    for c in title:
        #title_data.append[char_index_dict[c]]
        title_data.append(c)
        if c not in char_index_dict:
            char_index_dict[c] = counter
            index_char_dict[counter] = c
            #title_data.append[counter]
            counter += 1
    titles_data.append(title_data)
    
vocab_len = len(char_index_dict)    
titles_data_len = len(titles_data) 

titles_total_len = sum(map(lambda x : len(x), titles_data))
#total number of samples to be gathered
counter = titles_total_len - tarmac_len * titles_data_len

#numpy arrays to hold data
x = np.zeros((counter, tarmac_len, vocab_len), dtype=np.bool)
y = np.zeros((counter, vocab_len), dtype=np.bool)

#gather moving window of data
for title in titles_data:
    counter -= 1
    if counter < 0:
        break
    title_len = len(title)
    #fill in x
    for j in range(tarmac_len, title_len-1):
        for k, c in enumerate(title[j-tarmac_len : j]):
            x[counter-1, k, char_index_dict[c]] = 1
        #fill in the char to predict
        y[counter-1, char_index_dict[title[j+1]]] = 1
        
#x has shape (754149, 15, 442)
#y has shape (754149, 442)

## Create and train network
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

num_epochs = 2
batch_sz = 128

def create_model():
    model = keras.Sequential()
    #add LSTM layer 
    model.add(LSTM(128, input_shape=(tarmac_len, vocab_len)))
    
    model.add(Dense(vocab_len))
    model.add(Activation('softmax'))
    return model

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y, batch_size=batch_sz, epochs=num_epochs, verbose=1)

#####
#time for some funky predictions
stop_chars = ".!?"
max_sent_len = 20

#seed must be at least tarmac_len chars
#predict words given a model
def title_gen(model, seed):
    seed_len = len(seed)
    if seed_len < tarmac_len:
        print("Please supply a seed with length at least {}" % tarmac_len)
        return
    generated_sent = []
    generated_sent.extend(list(seed))
    sent = seed
    counter = max_sent_len
    c = ' '
    generated_sent.append(c)
    while(c not in stop_chars):
        x = np.zeros((1, tarmac_len, vocab_len))
        for i, c in enumerate(generated_sent[-tarmac_len : -1]):
            x[0, i, char_index_dict[c]] = 1
        #this returns a distribution
        probs = model.predict(x)[0]
        #treat these probs as a distribution and sample
        #without sampling, the model predictions are very monotonous,
        #e.g. "t" is over-predicted
        c_index = np.argmax(np.random.multinomial(1, probs))
        #print(c_index)
        c = index_char_dict[c_index]
        generated_sent.append(c)
        counter -= 1
        if counter < 0:
            break
    return "".join(generated_sent)
    
    
