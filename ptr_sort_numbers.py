import numpy as np, sys
from pickle import dump
from keras.models import Model, Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dropout, Dense, Activation, Input, Bidirectional
from keras.optimizers import Adam, SGD
from models import ptrnet_model
from scipy.stats import pearsonr

import argparse
parser = argparse.ArgumentParser(description='Number sorter')
parser.add_argument('-N', action='store', dest='N', default=10, type=int)
parser.add_argument('-m', action='store', dest='m', default=1, type=int)
parser.add_argument('--replace', action='store', dest='replace', default=False, type=bool)
parser.add_argument('-P', action='store', dest='P', default=2, type=int)
parser.add_argument('--batch-size', action='store', dest='batch_size', default=1, type=int)


args = parser.parse_args()

N = num_timesteps = args.N
hiddenUnits = N*10
max_num = N**args.P
P = args.P
batch_size = args.batch_size 

def one_hot(numbers, num_timesteps, max_num):
    x = np.zeros((len(numbers), num_timesteps, max_num), dtype=np.int)
    for ind, batch in enumerate(numbers):
        for j, number in enumerate(batch):
            x[ind, j, number] = 1
    return x

def batch_gen(batch_size=256, num_timesteps=10, max_num=100):
    x = np.zeros((batch_size, num_timesteps, max_num))
    y = np.zeros((batch_size, num_timesteps, max_num))

    while True:
        numbers = np.random.randint(max_num, size=(batch_size, num_timesteps))
        _y = np.random.randint(max_num, size=(batch_size, num_timesteps))
        sorted_numbers = np.sort(numbers, axis=1)

        for b in range(batch_size):
            data = np.vstack((sorted_numbers[b], range(num_timesteps))).T.copy()
            np.random.shuffle(data)
            np.random.shuffle(data)
            np.random.shuffle(data)
            numbers[b] = data[:,0]
            _y[b] = data[:,1]

        x = one_hot(numbers, N, max_num)
        y = one_hot(_y, N, N)

        yield x,y
        x.fill(0)
        y.fill(0)

def evaluate_loss(outputs, golden):
    loss = 0.0
    reverse_golden = np.array(range(len(outputs)))
    lossMax = np.sqrt(np.sum((golden - reverse_golden)**2))
    loss = np.sqrt(np.sum((golden-outputs)**2))

    loss /= lossMax
    return loss


def create_model_1():
    
    """ Bidirectional Encoder Sharing paramteres with Decoder. Parallel Decoding """

    enc_input = Input(shape=(num_timesteps, max_num))
    enc = Bidirectional(LSTM(hiddenUnits, return_sequences=True))(enc_input)
#    enc = Dropout(0.25)(enc)
    dec = Bidirectional(LSTM(hiddenUnits, return_sequences=True))(enc)
    dec = TimeDistributed(Dense(max_num, activation='softmax'))(dec)

    model = Model(input=[enc_input], output=[dec])
    return model

def create_model_5():
    
    """ Encoder with Attention to Decoder."""

    enc_input = Input(shape=(num_timesteps, max_num))
    enc = LSTM(hiddenUnits, return_sequences=True)(enc_input)
    dec = LSTM(hiddenUnits, return_sequences=True)(enc)
    dec = TimeDistributed(Dense(max_num, activation='softmax'))(dec)

    model = Model(input=[enc_input], output=[dec])
    return model


def create_model_2():

    """ Bidirectional Stacked Encoder, last timestep to Decoder """

    enc_input = Input(shape=(num_timesteps, max_num))
    enc = Bidirectional(LSTM(hiddenUnits, return_sequences=True))(enc_input)
    enc = Bidirectional(LSTM(hiddenUnits))(enc)
    enc = Dropout(0.25)(enc)
    enc = RepeatVector(num_timesteps)(enc)
    dec = Bidirectional(LSTM(hiddenUnits, return_sequences=True))(enc)
    dec = TimeDistributed(Dense(max_num, activation='softmax'))(dec)

    model = Model(input=[enc_input], output=[dec])
    return model


def create_model_3():

    """ Bidirectional Seq2Seq"""

    enc_input = Input(shape=(num_timesteps, max_num))
    enc = Bidirectional(LSTM(hiddenUnits))(enc_input)
#    enc = Dropout(0.25)(enc)
    enc = RepeatVector(num_timesteps)(enc)
    dec = Bidirectional(LSTM(hiddenUnits, return_sequences=True))(enc)
    dec = TimeDistributed(Dense(max_num, activation='softmax'))(dec)

    model = Model(input=[enc_input], output=[dec])
    return model

def create_model_4():

    """ LSTM Seq2Seq """

    enc_input = Input(shape=(num_timesteps, max_num))
    enc = LSTM(hiddenUnits)(enc_input)
    enc = Dropout(0.25)(enc)
    enc = RepeatVector(num_timesteps)(enc)
    dec = LSTM(hiddenUnits, return_sequences=True)(enc)
    dec = TimeDistributed(Dense(max_num, activation='softmax'))(dec)

    model = Model(input=[enc_input], output=[dec])
    return model


adam = Adam(lr=0.01, decay=0.0001)

#func = [create_model_1, create_model_2, create_model_3, create_model_4, create_model_5][int(sys.argv[1])-1]
#func = [create_model_1, create_model_2, create_model_3, create_model_4, create_model_5][args.m-1]

model = ptrnet_model(input_dim=max_num, timesteps=N, hiddenStates=N*10)
#model = func()

#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

loss_thresh = 0.15
#model.load_weights('shuff2seq.h5')
count = 0


history = {'loss':[], 'spear':[]}

for ind, (X,Y) in enumerate(batch_gen(batch_size, num_timesteps, max_num)):
    loss = model.train_on_batch(X,Y)
#    history['acc'].append(acc)

    if (ind+1)%50 == 0:
        history['loss'].append(loss)
        numbers = np.random.randint(max_num, size=(1, num_timesteps))
        _y = np.random.randint(max_num, size=(1, num_timesteps))
        sorted_numbers = np.sort(numbers, axis=1)

        b = 0
        data = np.vstack((sorted_numbers[b], range(num_timesteps))).T.copy()
        np.random.shuffle(data)
        numbers[b] = data[:,0]
        _y[b] = data[:,1]

        testX = one_hot(numbers, N, max_num)
        testY = one_hot(_y, N, N).squeeze()

        print ind+1, loss
        y = model.predict(testX, batch_size=1).squeeze()
        true1 = np.argsort(testY, axis=1)[:,-1]
        pred1 = np.argsort(y, axis=1)[:,-1]
        true2 = np.argsort(testY.T, axis=1)[:,-1]
        pred2 = np.argsort(y.T, axis=1)[:,-1]

        print pearsonr(true1, pred1)[0]
        _pear = max([pearsonr(true1, pred1)[0], pearsonr(true2, pred2)[0]])
        print "="*100

        history['spear'].append(_pear)

        if _pear > 0.99:
            break
dump(history,open('history_ptr_{0}_{1}.p'.format(N,P),'wb'))
