from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.layers import Merge,Dense,Input, LSTM, Bidirectional, TimeDistributed, Dropout
from pointer_net import PointerLSTM

from magic import make_parallel

def baseline_seq2seq_model(input_dim=8896, hiddenStates = 512):
    _input = Input(shape=(5,input_dim))
    encoder_context = Dropout(0.2)(LSTM(hiddenStates, return_sequences=True)(_input))

    decoder = Dropout(0.2)(LSTM(hiddenStates, return_sequences=True)(encoder_context))
    output = TimeDistributed(Dense(5, activation='softmax'))(decoder)
    model = Model(_input, output)
    adam = Adam(lr=0.01, decay=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model

def ptrnet_model(input_dim=8896, timesteps = 5,  hiddenStates = 128, parallel=False):
    _input = Input(shape=(timesteps,input_dim))
    encoder_context = Dropout(0.2)(LSTM(hiddenStates, return_sequences=True)(_input))
    decoder = PointerLSTM(hiddenStates, hiddenStates)(encoder_context)
    model = Model(_input,decoder)

    if parallel:
        model = make_parallel(model, 4)
    adam = Adam(lr=0.01, decay=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam)

    return model


