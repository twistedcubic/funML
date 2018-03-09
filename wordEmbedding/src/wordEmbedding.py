import sys
import codecs
import re
import collections
import numpy as np
import math
import random
from six.moves import xrange
import tensorflow as tf

data_index = 0
def generateBatch(data, batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [skip_window]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

lines = [line.rstrip('\n') for line in open('wordVecsSample.txt')]
path_to_data = 'path/to/data.txt'
linesIter = iter(lines)
next(linesIter)
wordDict = dict()

try:
  while True:
    line = next(linesIter)
    #separate first word and the ensuing vec
    lineTokens = line.split()
    word = lineTokens[0]
    wordVec = list(map(lambda x: float(x), lineTokens[1:]))
    wordDict[word]= wordVec;
except StopIteration:
  print 'Done with reading words from wordVecsSample.txt!'
  #print 'wordDict: ', wordDict

# strip punctuations, only use words
#read in the raw words data from file
def readFile(filename):
  data = list()
  for line in codecs.open(filename, encoding='utf-16'):
    #data.append(re.split('[\\s|,]', line))
    data.append(line.split()) #strip() doesn't work here??
  #flatten nested list (2 level deep)  
  return [e for sublist in data for e in sublist]

wordStrDataList = readFile(path_to_data)
#wordStrDataList = readFile('skipGramWordsList.txt')
print 'Done with forming wordStrDataList!'

#need to go through data twice, first time build the dict
#use counter to get the most frequent ones, for non frequent ones, can't get good vec anyway
#vocabSz = int(len(wordStrDataList)*3/4)
#embeddingSz = 300; #size of pretrained vecs

def buildData(wordStrDataList):
  countList = [('UNK', -1)]
  countMap = collections.Counter(wordStrDataList)
  print 'length of wordStrDataList: ', len(wordStrDataList)
  vocabSz = int(len(countMap)*8/24)
  countList.extend(countMap.most_common(vocabSz-1))

  print 'countList middle, 2/3, freq: ',countList[int(len(countList)*2/3)]
  print 'countList end, 10/11 freq: ',countList[int(len(countList)*10/11)]
  #dictionary of words and their indices, used to form data
  wordIndexMap = dict()
  for w, _ in countList:
    wordIndexMap[w] = len(wordIndexMap)
  
  reverseWordIndexMap = dict(zip(wordIndexMap.values(), wordIndexMap.keys()))
  
  wordIntDataList = list()
  for w in wordStrDataList:
    if w in wordIndexMap:
      wordIntDataList.append(wordIndexMap[w])
    else:
      #'UNK' is the word at the 0th index    
      wordIntDataList.append(0)
      
  return wordIntDataList, wordIndexMap, reverseWordIndexMap, vocabSz

wordIntDataList, wordIndexMap, reverseWordIndexMap, vocabSz = buildData(wordStrDataList)
print 'Built data such as wordIntDataList'
batchSz = 128
embeddingSz = 300
skipWindow = 2
numReuse = 4

def generateEmbeddingMx():
  initialEmbeddingsList = list()
  numPretrainedVecsFound = 0
  print 'in generatedEmbeddingMx'
  for i in range(len(wordIndexMap)):
    w = reverseWordIndexMap.get(i)
    #print 'w in generateEmbeddingMx: ', w
    if w in wordDict:          
      #print w,'in wordDict!'
      embeddingVec = wordDict.get(w)
      numPretrainedVecsFound += 1
      #print 'found word ', w, 'in wordDict'
    else:
      #initial weight vectors. Words correspond to rows.
      embeddingVec = tf.random_uniform([embeddingSz], -1.0, 1.0)
    initialEmbeddingsList.append(embeddingVec)
  print 'number of pretrained vectors found: ', numPretrainedVecsFound
  #defaults to packing along 0th dimension
  embeddingMx = tf.Variable(tf.pack(initialEmbeddingsList))
  return embeddingMx

#validation set
#validSz = 20
#select from middle of vocab in terms of frequencies
validWindow = np.arange(vocabSz/300+50, vocabSz/300+300)
validationWords = ["sheaf","subgroup","field","orthogonal","transverse","matrix","asymptotic","linear","commutative","abelian",'valence','graph','lagrangian','commutativity','associative','diffeomorphism','bifurcation','banach space','vertex','adjoint','distribution','polynomial','tangent','surjective','injective']
validExamples = []
#validationWordCounter = 0
for w in validationWords:
  if w in wordIndexMap:
    validExamples.append(wordIndexMap[w])
validSz = len(validExamples)
    #validationWordCounter += 1
#validExamples = np.random.choice(validWindow, validSz, replace=False)
numNegativeSampled = 64 #number of random negative context words to take in NCE

def printValidation():
  sim = similarity.eval()
  for i in range(validSz):
    validationWord = reverseWordIndexMap[validExamples[i]]
    
    print 'Similar words to: ', validationWord
    nearestNum = 12
    #this sorts through a vec of size vocabSz!
    nearestVecs = (-sim[i, :]).argsort()[1:nearestNum+1]
    for j in range(nearestNum):
      sys.stdout.write(reverseWordIndexMap[nearestVecs[j]])
      sys.stdout.write('  ')
    sys.stdout.write('\n')
  print '~~~~~Done validating for this round~~~~~~'
  
graph = tf.Graph()

with graph.as_default():
  trainInputs = tf.placeholder(tf.int32, shape=[batchSz])
  trainLabels = tf.placeholder(tf.int32, shape=[batchSz, 1])
  validDataset = tf.constant(validExamples, dtype=tf.int32)

  with tf.device('/cpu:0'):
    
    embeddings = generateEmbeddingMx()
    embed = tf.nn.embedding_lookup(embeddings, trainInputs)
    nceWeights = tf.Variable(tf.truncated_normal([vocabSz, embeddingSz], stddev=1.0/math.sqrt(embeddingSz)))
    nceBiases = tf.Variable(tf.zeros([vocabSz]))

    print 'Dimensions of: nceWeights: ', nceWeights.get_shape(), ' biases: ', nceBiases.get_shape(),' embeddings: ', embeddings.get_shape()

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nceWeights, biases=nceBiases, labels=trainLabels, inputs=embed,num_sampled=numNegativeSampled,num_classes=vocabSz))

    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
    #use dot product to compute similarity
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalizedEmbeddings = embeddings/norm
    validEmbeddings = tf.nn.embedding_lookup(normalizedEmbeddings, validDataset)
    similarity = tf.matmul(validEmbeddings, normalizedEmbeddings, transpose_b=True)
    init = tf.global_variables_initializer()

  #saver = tf.train.Saver({'embeddings':embeddings})  
  numSteps = 150001
  with tf.Session(graph=graph) as session:
    init.run()
    print 'Initialized'
    averageLoss = 0

    #tf.add_to_collection('train_op', [reverseWordIndexMap])
    for step in xrange(numSteps):
      batchInputs, batchLabels = generateBatch(wordIntDataList, batchSz, numReuse, skipWindow)
      feedDict = {trainInputs: batchInputs, trainLabels: batchLabels}

      _, lossVal = session.run([optimizer, loss], feed_dict=feedDict)
      averageLoss += lossVal

      if step % 10000 == 0:
        if step > 0:
          print 'average loss: ', averageLoss/3000
          averageLoss = 0
        print 'WEIGHTS: ', nceWeights.eval()
        printValidation()
    finalEmbeddings = normalizedEmbeddings.eval()
    #
    for w in validationWords:
      sys.stdout.write( w +' pretrained? ' + str(w in wordDict))      

    #save graph to file
    #saver.save(session, 'testWordModel')
