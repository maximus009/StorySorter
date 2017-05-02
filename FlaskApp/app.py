from flask import Flask, render_template,request
from flask_bootstrap import Bootstrap
import json
from main import get_stories,get_random
app = Flask(__name__)

store_id=0
@app.route("/",methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        album_id = int(request.form['album_id'])
        global store_id
        store_id=album_id
        input_image,input_text,output_image,output_text=get_stories(album_id)
        return render_template('index.html',sorted_image=output_image,sorted_text=output_text,shuffled_image=input_image,shuffled_text=input_text)
    else:
        input_image,input_text,output_image,output_text=get_stories(Random=True)
        return render_template('index.html',sorted_image=input_image,sorted_text=input_text,shuffled_image=output_image,shuffled_text=output_text)

@app.route("/shuffle",methods=['POST'])

def shuffle():
    album_id = store_id
    input_image,input_text,output_image,output_text=get_stories(album_id,shuffle=True)
    return render_template('index.html',sorted_image=output_image,sorted_text=output_text,shuffled_image=input_image,shuffled_text=input_text)
    

@app.route("/arrange",methods=['POST'])

def arrange():
    album_id = store_id
    input_image,input_text,output_image,output_text=get_stories(album_id,shuffle=False)
    return render_template('index.html',sorted_image=output_image,sorted_text=output_text,shuffled_image=input_image,shuffled_text=input_text)
    




@app.route("/example")
def example():
    input_image_list=[]
    input_text_list=[]
    output_image_list=[]
    output_text_list=[]
    index=[243,654,342,676,23]
    for i in index:
        input_image,input_text,output_image,output_text=get_random(i)
        input_image_list.append(input_image)
        input_text_list.append(input_text)
        output_image_list.append(output_image)
        output_text_list.append(output_text)
    return render_template('example.html',im1=input_image_list,text1=input_text_list,im2=output_image_list,text2=output_text_list)

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)