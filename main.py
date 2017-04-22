from data_utils import get_features, get_story, dump_pkl, load_pkl
from models import baseline_seq2seq_model, ptrnet_model 
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='Story sorter')
parser.add_argument('--no-image', dest='NO_IMAGE', default=False, action='store_true')
parser.add_argument('--no-text', dest='NO_TEXT', default=False, action='store_true')
parser.add_argument('--batch-size', dest='batch_size', default=False, action='store')

args = parser.parse_args()
batch_size = int(args.batch_size)
input_dim = 0

if args.NO_IMAGE and args.NO_TEXT:
    print 'You need at least one feature input'
    print 'Program exit.'
    exit()

elif args.NO_IMAGE:
    #only text
    input_dim = 4800
elif args.NO_TEXT:
    #only image
    input_dim = 4096
else:
    # default
    # both features
    input_dim = 4096+4800


np.set_printoptions(2)


def generate_batch(batch_size=1, training=True, return_image=True, return_text=True, mode='concat', shuffle=True):
    
    if training:
        randomIndex = np.random.choice(load_pkl('chosenIndices.p'), batch_size)
    else:
        randomIndex = np.random.randint(8*L/10, L,batch_size)
    x = np.zeros((batch_size * 120, 5,  input_dim)) 
    y = np.zeros((batch_size * 120, 5, 5))

    for i,index in enumerate(randomIndex):
        X, Y = get_story(index, input_dim, return_image=return_image, return_text=return_text, verbose=False)
        for index,(_x,_y) in enumerate(zip(X,Y)):
            x[i*120 + index] = _x
            y[i*120 + index] = _y
            
    return x,y


if __name__ == "__main__":
    x,y = generate_batch(100)
    print x.shape, y.shape
    model = ptrnet_model(input_dim = 8896, hiddenStates = 128)
    model.train_on_batch(x,y)
