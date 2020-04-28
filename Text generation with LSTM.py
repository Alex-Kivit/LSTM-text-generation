import os
import glob
import time
import random
import string
import tensorflow as tf
from numpy import array
from pickle import dump
from pickle import load
from tkinter import *
from tkinter import filedialog
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding, Dropout

#Configure GPUs for tensorflow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def load_doc(filename):
    # open the file
    file = open(filename, 'r', encoding='latin-1')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    # replace '--' with a space ' '
    doc = doc.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'a+')
    file.write(data)
    file.close()
    
# reduce length of incorrect lines to 11
def clean_sequences(sequences):
    count=0
    incorrect_lines = list()
    for i in sequences:
        if len(i)>11:
            while len(sequences[count])!=11:
                sequences[count].pop(-1)
        count+=1
    return sequences

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

def text2sequences(file):

    os.chdir(folder_selected)
    # load document
    doc = load_doc(file)

    # clean document
    tokens = clean_doc(doc)
    print(tokens[:200])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))

    # organize into sequences of tokens
    length = 10 + 1
    sequences = list()
    for i in range(length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i-length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))

    # save sequences to file
    os.chdir(home_directory)
    save_doc(sequences, 'sequences.txt')




# remove "sequences.txt" if it exists
if os.path.exists('sequences.txt'):
    os.remove('sequences.txt')

# select directory with texts
root = Tk()
root.withdraw()
print('Please select the folder with texts...')
time.sleep(3)
folder_selected = filedialog.askdirectory()

home_directory = os.getcwd()
os.chdir(folder_selected)
files = glob.glob('*.txt')

for file in files:
    text2sequences(file)
    
# load sequences
doc = load_doc('sequences.txt')
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

sequences = array(clean_sequences(sequences))
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define and train the model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=128, epochs=100)

model.save('model.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))
 
# load cleaned text sequences
doc = load_doc('sequences.txt')
lines = doc.split('\n')
seq_length = len(lines[0].split()) - 1

model = load_model('model.h5')
tokenizer = load(open('tokenizer.pkl', 'rb'))

# get random "seed" sequence
seed_text = lines[random.randint(0,len(lines))]
print(seed_text + '\n')

# generate text
generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
print(generated)
