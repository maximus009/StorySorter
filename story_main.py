from data_utils import get_features, get_story, dump_pkl, load_pkl
from models import baseline_seq2seq_model, ptrnet_model 
import argparse
import numpy as np
from time import time


parser = argparse.ArgumentParser(description='Story sorter')
parser.add_argument('--no-image', dest='NO_IMAGE', default=False, action='store_true')
parser.add_argument('--no-text', dest='NO_TEXT', default=False, action='store_true')
parser.add_argument('--batch-size', dest='batch_size', default=512, action='store', type=int)
parser.add_argument('--nb-epoch', dest='nb_epoch', default=1, action='store', type=int)
parser.add_argument('-K', dest='K', default=5, action='store', type=int)

args = parser.parse_args()
batch_size = args.batch_size
nb_epoch = args.nb_epoch
input_dim = 0

return_text, return_image = True, True
if args.NO_IMAGE and args.NO_TEXT:
    print 'You need at least one feature input'
    print 'Program exit.'
    exit()

elif args.NO_IMAGE:
    #only text
    return_image = False
    input_dim = 4800

elif args.NO_TEXT:
    #only image
    return_text = False
    input_dim = 4096
else:
    # default
    # both features
    input_dim = 4096+4800


np.set_printoptions(2)

trainIndices = load_pkl('trainIndices')
testIndices = load_pkl('testIndices')


def generate_batches(batch_size=1, K=4, training=True, return_image=return_image, return_text=return_text):
    

    if training:
        _indices = trainIndices
        _run_mode = 'train'
    else:
        _run_mode = 'test'
        _indices = testIndices

    num_of_batches = len(_indices)/batch_size

    print "Generating {0} {1} batches".format(num_of_batches, _run_mode)

    randomIndices = []
    for b in range(num_of_batches):
        print '{2} batch: {0}/{1}'.format(b+1,num_of_batches, _run_mode)
        randomIndex = np.random.choice(_indices,batch_size)

        x = np.zeros((batch_size * K, 5,  input_dim)) 
        y = np.zeros((batch_size * K, 5, 5))

        for i,r_index in enumerate(randomIndex):
            X, Y = get_story(r_index, input_dim, K, return_image=return_image, return_text=return_text, verbose=False)
            x[i*K:(i+1)*K,:] = X
            y[i*K:(i+1)*K,:] = Y
                    
        yield x,y

def train():
    expName = 'ptr128_image_{0}_text_{1}'.format(str(return_image),str(return_text))
    model = ptrnet_model(input_dim = input_dim, hiddenStates = 128)
    loss = []
    print input_dim
    min_loss = 10
    for ep in range(nb_epoch):
        ep_start = time()
        print "Epoch:",ep+1
        _loss = 0.0
        for b, (x,y) in enumerate(generate_batches(batch_size)):
            h = model.fit(x,y, batch_size=4096*4, verbose=0, nb_epoch=1)
            _loss += h.history['loss'][0]
        print "\n",_loss/b

        test_loss = 0.0
        for b, (x,y) in enumerate(generate_batches(batch_size, training=False)):
            test_loss += model.evaluate(x,y, verbose=0, batch_size=4096*4)
        test_loss /= b
        print 'test loss:', test_loss

        
        if test_loss < min_loss:
            print 'Loss improved from {0} to {1}'.format(min_loss, test_loss)
            min_loss = test_loss
            print 'Saving model_%s' % expName
            model.save('model_%s.h5' % expName)
            model.save_weights('weights_%s.h5' % expName)
        print "="*100
        loss.append(_loss)
        print time() - ep_start, "seconds for epoch",ep+1

if __name__ == "__main__":
    train()
