import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


import random

#tensorboard
%load_ext tensorboard
from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import numpy as np


words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
X = list(training[:,0])
y = list(training[:,1])

print("Training data created")

#log tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) #tambah
model.add(Dense(len(y[0]), activation='softmax'))


# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.summary()

#fitting and saving the model
#hist = model.fit(np.array(X), np.array(y), epochs=200, batch_size=5, verbose=1, 
                 #validation_data=(X, y), callbacks=[tensorboard_callback],)


from sklearn.model_selection import train_test_split
#train_size = 0.8
X_train, X_rem, y_train, y_rem = train_test_split(
    X,y, train_size=0.7)
#test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(
    X_rem,y_rem, test_size=0.4)


hist = model.fit(np.array(X_train), np.array(y_train), epochs=300, batch_size=5, verbose=1,
                 validation_data=(X_valid, y_valid),
                 callbacks=[tensorboard_callback],)




#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) #shuffle=1)

#hist = model.fit(np.array(X_train), np.array(y_train), epochs=300, batch_size=5, verbose=1, 
                 #validation_data=(X_test, y_test), callbacks=[tensorboard_callback],)



#hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1, 
                 #validation_data=(train_x, train_y), callbacks=[tensorboard_callback],)




#Save model h5 to drive
model.save('chatbot_model.h5', hist) 


print("model created")


from flask import Flask, jsonify
import os

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import time

## end keras chat brain

## vars
now = time.time()# float

filename = str(now)+"_chatlog.txt" #create chatlog

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
    
## end vars

class Storage:
    old_answers=[] #storage for answers
    
    @classmethod
    def save_storage(cls):
        with open ("storage.txt", "w") as myfile:
            for answer in Storage.old_answers:
                
                myfile.write(answer+"\n")

    @classmethod
    def load_storage(cls):
        Storage.old_answers=[]
        with open ('storage.txt', 'r') as myfile:
            lines = myfile.readlines()
            for line in lines:
                Storage.old_answers.append(line.strip())
        print (Storage.old_answers)


app = Flask(__name__)
run_with_ngrok(app) 

def bot_response(userText):

    '''fake brain'''
    print ("your q was: " + userText)
    return "your q was: " + userText
   
## new funcs
def clean_up_sentence(sentence):
    """tokenizes the sentences"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    '''read in the intents file'''
    #pseudo code
    #assume old answers are inside
    # old_answers = ['response1','response2']

    #load old answers into storage
    Storage.load_storage()
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    old_answers = Storage.old_answers  # [:-len(list_of_intents)]
    possible_responses = [i['responses'] for i in list_of_intents if i['tag']== tag ][0]
    history = Storage.old_answers[-len('possible_responses'):]
    print("** possible answers and history old answers",possible_responses,history)
    unused_answers = [answer for answer in possible_responses if answer not in history ] # list comprehension
    print(unused_answers, " unused answers")
    unused_two = history[-(len(possible_responses)-1):]
    print(unused_two,'last five answers')
    try:
        result = random.choice([answer for answer in possible_responses if answer not in unused_two ])
    except IndexError:
        print("I'm out of options, I will choose random.")
        result = random.choice(possible_responses)

    Storage.old_answers.append(result) 
    Storage.old_answers= Storage.old_answers[-20:] 
    Storage.save_storage()

  
    return result,tag

def chatbot_response(msg):
    '''this func is important'''
    ints = predict_class(msg, model)
    res,tag = getResponse(ints, intents)
    return res,tag


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        #append to log file
        with open(filename,'a') as myfile:
            myfile.write("user: "+ msg + "\n")
            myfile.write("bot: "+ res + "\n")

## end new funcs

@app.route('/')
def home():
    return render_template('home.html')

#
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route("/get")
def get_bot_response():    
    print ("get is called")
    userText = request.args.get('msg')    
    # return str(bot.get_response(userText)) 
    # return bot_response(userText)
    res,tag = chatbot_response(userText)
    with open( "/content/drive/MyDrive/Colab Notebooks/Chatbot/logfile.csv", "a" ) as logfile:
        logfile.write(str(now)+","+userText+","+res+","+tag+","+"\n")
        

    return res + '<p style="font-size:8pt;">tag: ' + tag + '</p>'



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
