"""
Program to sort shuffed/jumbled words in a sentence
"""
from keras.datasets.imdb import load_data
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Dense, TimeDistributed, LSTM, Bidirectional, \
                        Input, Embedding

from pointer_net import PointerLSTM
from scipy.stats import pearsonr, spearmanr
from magic import make_parallel
                        
from data_utils import load_pkl, dump_pkl
import numpy as np
from sys import exit

if False:
    (x_train, y_train), (x_test, y_test) = load_data(path="imdb.npz",
                                                          skip_top=0,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=0,
                                                          index_from=2)
    n = 10
    n_grams = []
    for x in x_train:
        l = len(x)
        for i in range(0,l-n+1,1):
            n_grams.append(x[i:i+n])
    n_grams = np.array(n_grams)
    dump_pkl(n_grams, 'sentences_10_grams')

else:
    sentences = load_pkl('sentences_10_grams')
    L = len(sentences)
    maxWordIndex =  np.max(sentences)

print L
def create_data(sentences, K=120):

    x = np.zeros((len(sentences), 10, 1))
    y = np.tile(np.eye(10), (len(sentences),1,1))

    x = sentences

    X = np.zeros((len(sentences)*K, 10, 1), dtype=np.int)
    Y = np.zeros((len(sentences)*K, 10, 10))

    for i,(_x,_y) in enumerate(zip(x,y)):
        x_y = np.column_stack((_x,_y)).copy()
        _X, _Y = x_y[:,:1], x_y[:,1:]
        X[i*K] = _X
        Y[i*K] = _Y
        for k in range(1,K):
            np.random.shuffle(x_y)
            _X, _Y = x_y[:,:1], x_y[:,1:]
            X[i*K+k] = _X
            Y[i*K+k] = _Y

    X = X.squeeze()
    return X,Y


def create_model():

    hiddenStates = 64 
    _input = Input(shape=(10,))
    sequence = Embedding(input_dim=maxWordIndex+1, output_dim=100, input_length=10)(_input)
    encoder = LSTM(hiddenStates, return_sequences=True)(sequence)
#    decoder = LSTM(hiddenStates, return_sequences=True)(encoder)
#    decoder = TimeDistributed(Dense(10, activation='softmax'))(decoder)

    decoder = PointerLSTM(hiddenStates, hiddenStates)(encoder)
    model = Model(_input,decoder)
    model = make_parallel(model, 4)
    adam = Adam(lr=0.1, decay=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model


model = create_model()

print model.summary()
trainSentences = sentences[:8*L/10]
testSentences = sentences[8*L/10:]
nb_epoch = 20
samples = 100000
min_loss = 3
losses = []

from time import time

for ep in range(nb_epoch):
    ep_time = time()
    for i in range(0,samples*5, samples):
        trainX, trainY = create_data(trainSentences[i:i+samples], K=4)
        h = model.fit(trainX, trainY, nb_epoch=1, batch_size=samples, shuffle=True, verbose=0)
        losses.append(h.history['loss'][0])
    dump_pkl(losses,'linearization_losses')
    inds = np.random.choice(len(testSentences), samples)
    testX, testY = create_data(testSentences[inds], K=4)
    loss = model.evaluate(testX, testY, batch_size=samples, verbose=0)
    print time() - ep_time,"seconds"
    if loss<min_loss:
        print "Loss improved from {0} to {1}.".format(min_loss, loss)
        model.save('lin_word.h5')
        min_loss = loss
    sampleX, sampleY = create_data(testSentences[np.random.randint(0,len(testSentences),1)], K=4)
    predictY = model.predict(sampleX,verbose=0).squeeze()[0]

    sampleY = sampleY[0]

    true1 = np.argsort(sampleY, axis=1)[:,-1]+1
    pred1 = np.argsort(predictY, axis=1)[:,-1]+1
    true2 = np.argsort(sampleY.T, axis=1)[:,-1]+1
    pred2 = np.argsort(predictY.T, axis=1)[:,-1]+1

    print true2, pred2
    print true1, pred1

    _pear2 = pearsonr(true2, pred2)
    _pear1 = pearsonr(true1, pred1)
    print _pear1, _pear2


#    _spear1 = spearmanr(true1, pred1)
#    _spear2 = spearmanr(true2, pred2)
#    print _spear1, _spear2
