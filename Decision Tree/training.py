import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('data.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        documents.append((w, intent['tag']))
        
        if (intent['tag'] not in classes):
            classes.append(intent['tag'])
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)

pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open("labels.pkl","wb"))

training = []

output_empty = [0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])
    
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


model = DecisionTreeClassifier(criterion="gini")
model.fit(np.array(train_x), np.array(train_y))
joblib.dump(model, 'DecisionTree.pkl')
print("model created")


    