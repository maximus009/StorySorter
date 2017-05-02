import json
import os
import random
import numpy as np
from time import sleep
from models import  ptrnet_model
from data_utils import get_features, get_story
from story_main import test
print get_features
print 'function loaded'

script_dir = os.path.dirname(__file__)
rel_path = "pred_file.json"
abs_file_path = os.path.join(script_dir, rel_path)
data=json.load(open(abs_file_path))

#model = train(True)
#expName = 'dev128_image_True_text_True'
#model.load_weights('weights_%s.h5' % expName)


val=[]
count=0
for d in data:
    d['index'] = count
    count=count+1

def get_stories(index=None, Random=False,shuffle=False):
#    sleep(1)
    get_list=[]
    img=[]
    text=[]
    output_image=[]
    output_text=[]
    m_index=[]
    t_index=[]
    v_index=[]
    if Random==True:
        index=np.random.randint(40066,size=1)
    for d in data:
        if index== int(d['index']):
            get_list=d['items']
            m_index= range(5)#i['m_pred_order']
            break
#            t_index=i['t_pred_order']
##           v_index=i['v_pred_order']
    input_index=[0,1,2,3,4]
    if shuffle:
        random.shuffle(input_index)
    for i in input_index:
        img.append(get_list[i]['url'])
        text.append(get_list[i]['text'])

    print input_index
    m_index = test(index=d['index'], input_index=input_index)
    print m_index 
#    inputStory, _ = get_story(d['index'], input_dim=8896, K=4)
#    inputStory[0] = inputStory[0][input_index]
#    features = get_features(story_id=d['story_id'], items=d)
#    print features[0][:5]
#    output = model.predict(inputStory, batch_size=1).squeeze()[0]

    m_index=[m-1 for m in m_index]
    t_index=[m-1 for m in t_index]
    v_index=[m-1 for m in v_index]
    for i in m_index:
        output_image.append(get_list[i]['url'])
        output_text.append(str(i+1)+'. '+get_list[i]['text'])
    return img,text,output_image,output_text

def get_random(id):
    input_img=[]
    input_text=[]
    img=[]
    text=[]
    index=id
    for d in data:
        if index == int(d['index']):
            get_list=d['items']
    for i in get_list:
        img.append(i['url'])
        text.append(i['text'])

    input_index=[4,1,3,2,0]
    #random.shuffle(input_index)
    for i in input_index:
        input_img.append(get_list[i]['url'])
        input_text.append(get_list[i]['text'])
        
    return input_img,input_text,img,text
