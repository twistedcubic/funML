
'''
Utilities functions
'''

'''                                                                                                                                                                    Ll
Load pretrained vectors, assuming each line has form "word v1 v2 ..."                                                                                                    
Returns:                                                                                                                                                                 
-dictionary of word key and np embedding vector.                                                                                                                         
'''
def load_pretrained(path):
    word_to_vec = {}
    with open(path, "r") as file:
        for line in file:
            line_ar = line.split()
            word = line_ar[0]
	    embed = np.asarray(line_ar[1:], dtype='float32')
            word_to_vec[word] = embed
    return word_to_vec

