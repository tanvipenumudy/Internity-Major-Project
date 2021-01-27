import nltk
#nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import joblib
import numpy as np

from nltk.corpus import stopwords
ignore_words = set(stopwords.words('english'))


model = joblib.load('RandomForest.pkl')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word=="not" or word not in ignore_words]
    return sentence_words
    
    
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if(w==s):
                bag[i] = 1
    return(np.array(bag))
    

def predict_class(sentence, model):
    
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    res = res.tolist()
    #print(classes[res.index(1)])
    return classes[res.index(1)]
    
def getResponse(ints, intents_json):
    # print(ints)
    tag = ints
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result
    
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    #print(res)
    return res

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)
    
if __name__ == "__main__":
    app.run()
    