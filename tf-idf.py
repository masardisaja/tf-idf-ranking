from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
from os import path

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math
import sys

# %load_ext autotime


title = "stories"
alpha = 0.3

# User input
corpus_dir = input('Corpus folder : ')
query = input('Search : ')

# Cek apakah corpus_dir exist
if not path.exists(corpus_dir):
    print('Folder tidak ditemukan.')
    sys.exit()
else:
    print('Melakukan pencarian di dalam corpus folder ' + str(os.getcwd())+ '/' + corpus_dir +'/.\n')
    print('Mohon menunggu...\n')
    #sys.exit()

# Proses data source

folders = [x[0] for x in os.walk(str(os.getcwd())+'/stories/')]

folders[0] = folders[0][:len(folders[0])-1]

#print(folders)


dataset = []

c = False

for i in folders:
    file = open(i+"/index.html", 'r')
    text = file.read().strip()
    file.close()

    file_name = re.findall('><A HREF="(.*)">', text)
    file_title = re.findall('<BR><TD> (.*)\n', text)

    if c == False:
        file_name = file_name[2:]
        c = True
        
#    print(len(file_name), len(file_title))

    for j in range(len(file_name)):
        dataset.append((str(i) +"/"+ str(file_name[j]), file_title[j]))
        
N = len (dataset)

def print_doc(id):
    print(dataset[id])
    file = open(dataset[id][0], 'r', encoding='cp1250')
    text = file.read().strip()
    file.close()
    print(text)
    
def convert_lower_case(data):
    print('---> Converting to lower case ')
    return np.char.lower(data)

def remove_stop_words(data):
    print('---> Removing stop words ')
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
    
def remove_punctuation(data):
    print('---> Removing punctuation ')
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
    
def remove_apostrophe(data):
    print('---> Removing apostrophe ')
    return np.char.replace(data, "'", "")
    
def stemming(data):
    print('---> Stemming ')
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
    
def convert_numbers(data):
    print('---> Convert numbers ')
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
    
def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data
    
### PROCESSING DATA ###

#print(*dataset, sep = '\n')
#sys.exit()

processed_text = []
processed_title = []

# dataset[n] = (file_path, title)
# Maka :
# dataset[n][0] = file_path
# dataset[n][1] = title
# Selanjutnya dilakukan preprocessing dan tokenize terhadap text body/content dan terhadap text title.
# Untuk preprocessing terhadap text body dilakukan dengan membaca setiap file pada file_path (dataset[n][0]).
# Untuk preprocessing terhadap text title dilakukan dengan memproses setiap title value (dataset[n][1]).

print('-> Preprocessing documents...\n')
nu = 0
for i in dataset[:N]:
    nu+=1
    print("["+str(nu)+"/"+str(N)+"] ")
    # Baca isi file pada file_path (dataset[n][0]) --> i[0])
    file = open(i[0], 'r', encoding="utf8", errors='ignore')
    text = file.read().strip() # Value text body berdasarkan hasil membaca isi file
    file.close()
    
    # Preprocessing dan tokenize text body
    processed_text.append(word_tokenize(str(preprocess(text))))
    # Preprocessing dan tokenize text title
    processed_title.append(word_tokenize(str(preprocess(i[1]))))
    print("\n")
    

# Buat dataset DF (Document Frequency) dari hasil tokenize    
DF = {}

for i in range(N):
    tokens = processed_text[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    tokens = processed_title[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
            
print('-> Library DF berhasil dibuat..\n')
#sys.exit()

for i in DF:
    DF[i] = len(DF[i])
    
# DF

total_vocab_size = len(DF)

total_vocab = [x for x in DF]

# print(total_vocab[:20])

def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c
    
doc = 0

tf_idf = {}

for i in range(N):
    
    tokens = processed_text[i]
    
    counter = Counter(tokens + processed_title[i])
    words_count = len(tokens + processed_title[i])
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1))
        
        tf_idf[doc, token] = tf*idf

    doc += 1
    
# tf_idf
print('-> Library TF_IDF text body berhasil dibuat..\n')

doc = 0

tf_idf_title = {}

for i in range(N):
    
    tokens = processed_title[i]
    counter = Counter(tokens + processed_text[i])
    words_count = len(tokens + processed_text[i])

    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values
        
        tf_idf_title[doc, token] = tf*idf

    doc += 1
    
# tf_idf_title
print('-> Library TF_IDF text title berhasil dibuat..\n')


for i in tf_idf:
    tf_idf[i] *= alpha
    
for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]
    

len(tf_idf)


# Fungsi Matching Score
def matching_score(k, query):
    print("######### Matching Score #########")
    print("\nQuery:", query)
    print("\n")
    
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))

    print("")
    print(tokens)
    print('\n')
    
    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")
    
    l = []
    
    for i in query_weights[:10]:
        l.append(i[0])
    
    print(l)
    
def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
    
D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

def gen_vector(tokens):

    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)

    query_weights = {}
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q

def cosine_similarity(k, query):
    print("\n\n######### Cosine Similarity #########")
    print("\nQuery:", query)
    print("\n")
    
    preprocessed_query = preprocess(query)
    tokens = word_tokenize(str(preprocessed_query))
    
    print("")
    print("Tokens : ", tokens)
    print('\n')
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
        
    out = np.array(d_cosines).argsort()[-k:][::-1]
    
    print("")
    
    print(out)

if len(query)>0:
    matching_score(10,query)
    cosine_similarity(10,query)
