import json
import os
import random
import numpy as np
from time import sleep

script_dir = os.path.dirname(__file__)
rel_path = "pred_file.json"
abs_file_path = os.path.join(script_dir, rel_path)
data=json.load(open(abs_file_path))

val=[]
count=0
for d in data:
    d['index'] = count
    count=count+1
def get_stories(index=None, Random=False,shuffle=False):
    sleep(1)
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
    for i in data:
        if index== int(i['index']):
            get_list=i['items']
            m_index=i['m_pred_order']
            t_index=i['t_pred_order']
            v_index=i['v_pred_order']
    input_index=[0,1,2,3,4]
    if shuffle:
        random.shuffle(input_index)
    for i in input_index:
        img.append(get_list[i]['url'])
        text.append(get_list[i]['text'])
    m_index=[m-1 for m in m_index]
    t_index=[m-1 for m in t_index]
    v_index=[m-1 for m in v_index]
    for i in m_index:
        output_image.append(get_list[i]['url'])
        output_text.append(get_list[i]['text'])
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
