import numpy as np, sys
from pickle import dump
from keras.models import Model, Sequential
from keras.layers import LSTM, TimeDistributed, RepeatVector, Dropout, Dense, Activation, Input, Bidirectional
from keras.optimizers import Adam, SGD

import argparse
parser = argparse.ArgumentParser(description='Number sorter')
parser.add_argument('-N', action='store', dest='N', default=10, type=int)
parser.add_argument('-m', action='store', dest='m', default=1, type=int)
parser.add_argument('--replace', action='store', dest='replace', default=False, type=bool)
parser.add_argument('-P', action='store', dest='P', default=2, type=int)

args = parser.parse_args()

N = num_timesteps = args.N
hiddenUnits = N*10
max_num = N**args.P
batch_size = 256

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
        sorted_numbers = np.sort(numbers, axis=1)

        x = one_hot(numbers, N, max_num)
        y = one_hot(sorted_numbers, N, max_num)

        yield x,y
        x.fill(0)
        y.fill(0)

def evaluate_loss(outputs, golden):
    loss = 0.0
    reverse_golden = np.array(golden.tolist()[::-1])
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
func = [create_model_1, create_model_2, create_model_3, create_model_4, create_model_5][args.m-1]
model = func()

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

loss_thresh = 0.15
#model.load_weights('shuff2seq.h5')
count = 0


history = {'loss':[], 'acc':[], 'test_loss':[]}

for ind, (X,Y) in enumerate(batch_gen(batch_size, num_timesteps, max_num)):
    loss, acc = model.train_on_batch(X,Y)
    history['loss'].append(loss)
    history['acc'].append(acc)

    if (ind+1)%1 == 0:
        testX = np.expand_dims(np.random.choice(max_num, num_timesteps, replace=args.replace), axis=0)
        print ind+1,
        print loss, acc
        test = one_hot(testX, num_timesteps, max_num)
        y = model.predict(test, batch_size=1)
        golden = np.sort(testX).ravel()
        outputs = np.argmax(y, axis=2).ravel()
        print golden.tolist()
        print outputs.tolist()

        _loss = evaluate_loss(outputs, golden)
        print _loss

        history['test_loss'].append(_loss)
        if _loss  == 0:
            if count==5:
               break 
            else:
                count += 1
        else:
            count=0
dump(history,open('history.p','wb'))
