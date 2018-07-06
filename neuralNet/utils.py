
'''
Utilities functions
'''
import os
import numpy as np

'''                                                                                                                                                                    Ll
Load pretrained vectors, assuming each line has form "word v1 v2 ..."                                                                                                    
Returns:                                                                                                                                                                 
-dictionary of word key and np embedding vector.                                                                                                                         
'''
def load_pretrained(path=os.path.join(os.path.dirname(__file__), "embed/glove.6b.50d.txt")):
    word_vecs = {}
    with open(path, "r") as file:
        for line in file:
            line_ar = line.split()
            word = line_ar[0]
	    embed = np.asarray(line_ar[1:], dtype='float32')
            word_vecs[word] = embed
    return word_vecs

if __name__=="__main__":
    word_vecs = load_pretrained()
    print word_vecs["the"]
