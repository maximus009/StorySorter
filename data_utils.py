from json import load
import numpy as np
import pickle

dict_file = load(open('dict_file.json'))
L = len(dict_file)


def load_pkl(fileName):
    pklData = pickle.load(open(fileName+'.p'))
    return pklData

def dump_pkl(pklData, fileName):
    pickle.dump(pklData, open(fileName+'.p','wb'))

def get_features(story_id, items, return_image=True, return_text=True, mode='concat'):
    if return_image:    
        img_features = np.load("../stories/gautam/data/vggFeatures/"+story_id+"_vgg.npy")

    if return_text:
        text_features = np.load("../stories/gautam/data/skipFeatures/"+story_id+"_skip.npy")
        
    if return_text and return_image:
        # 'concat' is the only mode
        return_list = np.hstack((img_features,text_features))
    
    else:
        if return_image:
            return_list = img_features
        else:
            return_list = text_features
            
    return return_list



def get_story(index=0, input_dim=0, K=4, return_image=True, return_text=True, shuffle=False, verbose=True):

    X = np.zeros((K,5,input_dim))
    Y = np.zeros((K,5,5))
    if verbose:
        print 'Fetching story:',index
    story = dict_file[index]
    items = story['items']
    features = get_features(story['story_id'], items, return_image = return_image, return_text = return_text)

    data = np.column_stack((features, np.eye(5))).copy()

    # appending the sorted as input
    X[0] = data[:,:-5]
    Y[0] = data[:,-5:]

    for k in range(1, K):
        np.random.shuffle(data)
        X[k] = data[:,:-5]
        Y[k] = data[:,-5:]

    return X,Y

def create_indices():
    indices = np.arange(L)
    np.random.shuffle(indices)
    trains, tests = np.split(indices,[8*L/10])
    dump_pkl(trains, 'trainIndices')
    dump_pkl(tests, 'testIndices')


if __name__ == "__main__":
    create_indices()
