from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import nltk
import numpy as np
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Charger le modèle
model = load_model('chatbot_model.h5')

# Charger les fichiers pickle
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Charger les intentions
intents = json.loads(open('intents.json').read())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.get_json().get("message")
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({"response": res})

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

if __name__ == "__main__":
    app.run(debug=True)
