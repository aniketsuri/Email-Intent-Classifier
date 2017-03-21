import numpy as np
import re
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


train_pos = []
train_neg = []
fname = 'enron_train.txt'
with open(fname) as f:
    train_set = f.readlines()
for i in range(len(train_set)):
    if train_set[i][0] == 'N':
        str1 = train_set[i][3:].strip().strip('.')
        str1 = re.sub(r'pm',r' time ',str1)
        str1 = re.sub(r'[0-9]+:[0-9]+',r' time ',str1)
        str1 = re.sub(r'[$]+[0-9]+', ' money ',str1)
        str1 = re.sub(r'[0-9]+',r' number ',str1)
        str1 = re.sub(r'(http|https)://[^\s]*', r' httpaddr ',str1)
        str1 = re.sub(r'[$]+', ' dollar ',str1)
        str1 = re.sub(r'[^\s]+@[^\s]+', 'emailaddr',str1)        
        str1 = re.sub(r'[^\w\s]', '',str1)
        str1 = str1.lower() 
        train_neg.append(str1)
    else:
        str1 = train_set[i][4:].strip().strip('.')
        str1 = re.sub(r'pm',r' time ',str1)
        str1 = re.sub(r'[0-9]+:[0-9]+',r' time ',str1)
        str1 = re.sub(r'[$]+[0-9]+', ' money ',str1)
        str1 = re.sub(r'[0-9]+',r' number ',str1)
        str1 = re.sub(r'(http|https)://[^\s]*', r' httpaddr ',str1)
        str1 = re.sub(r'[$]+', ' dollar ',str1)
        str1 = re.sub(r'[^\s]+@[^\s]+', 'emailaddr',str1)        
        str1 = re.sub(r'[^\w\s]', '',str1) 
        str1 = str1.lower() 
        train_pos.append(str1)


test_pos = []
test_neg = []
fname = 'enron_test.txt'
with open(fname) as f:
    test_set = f.readlines()
for i in range(len(test_set)):
    if test_set[i][0] == 'N':
        str1 = test_set[i][3:].strip().strip('.')
        str1 = re.sub(r'pm',r' time ',str1)
        str1 = re.sub(r'[0-9]+:[0-9]+',r' time ',str1)
        str1 = re.sub(r'[$]+[0-9]+', ' money ',str1)
        str1 = re.sub(r'[0-9]+',r' number ',str1)
        str1 = re.sub(r'(http|https)://[^\s]*', r' httpaddr ',str1)
        str1 = re.sub(r'[$]+', ' dollar ',str1)
        str1 = re.sub(r'[^\s]+@[^\s]+', 'emailaddr',str1)        
        str1 = re.sub(r'[^\w\s]', ' ',str1) 
        str1 = str1.lower() 
        test_neg.append(str1)
    else:
        str1 = test_set[i][3:].strip().strip('.')
        str1 = re.sub(r'pm',r' time ',str1)
        str1 = re.sub(r'[0-9]+:[0-9]+',r' time ',str1)
        str1 = re.sub(r'[$]+[0-9]+', ' money ',str1)
        str1 = re.sub(r'[0-9]+',r' number ',str1)
        str1 = re.sub(r'(http|https)://[^\s]*', r' httpaddr ',str1)
        str1 = re.sub(r'[$]+', ' dollar ',str1)
        str1 = re.sub(r'[^\s]+@[^\s]+', 'emailaddr',str1)        
        str1 = re.sub(r'[^\w\s]', ' ',str1) 
        str1 = str1.lower() 
        test_pos.append(str1)

#print test_pos[:5]
#print test_neg[:5]

document_train = []
for line in train_pos:
    document_train.append((line.split(),'pos'))
for line in train_neg:
    document_train.append((line.split(),'neg'))

document_test = []
for line in test_pos:
    document_test.append((line.split(),'pos'))
for line in test_neg:
    document_test.append((line.split(),'neg'))

random.shuffle(document_train)
random.shuffle(document_test)


train_pos_words = [word for line in train_pos for word in line.strip().split()]
train_neg_words = [word for line in train_neg for word in line.strip().split()]
test_pos_words = [word for line in test_pos for word in line.strip().split()]
test_neg_words = [word for line in test_neg for word in line.strip().split()]
wordList = train_pos_words+train_neg_words+test_pos_words+test_neg_words


stop_words_ = [
'a','above','again','against','am','an','are','and','for','in','if','or','as','is'
'arent','because','been','being','below',
'between','both','but','by','during','each',
'few','further','had','had','has','hasnt','have','havent',
'having','he','hed','hell','hes','her','here','heres',
'hers','herself','him','himself','his','i','ill','im',
'ive','into','isnt','it','its','its','itself','me',
'more','my','myself','no','nor','of','off',
'on','once','only','other','our','ours','ourselves','out',
'over','own','same','shant','she','shed','she','she',
'so','some','such','than','that','thats','the','their',
'theirs','them','themselves','then','there','theres','these','they',
'theyd','theyll','theyre','theyve','this','those','through','to',
'too','under','up','very','was','wasnt','we',
'wed','well','were','weve','were','werent','what','whats',
'where','wheres','which','who','whos','whom','why','whys',
'youd','youll','youre','youve','your','yours','yourself','yourselves',]




unique_wordList = []
all_words_except_stop_words = []
for item in wordList:
  if item not in stop_words_ and len(item) > 1:
    all_words_except_stop_words.append(item)

all_words_except_stop_words = nltk.FreqDist(all_words_except_stop_words)
print "\nMost common 30 words are : ", all_words_except_stop_words.most_common(30)

word_features = [tup[0] for tup in all_words_except_stop_words.most_common(500)]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
#print((find_features(train_pos[0].split())))

trainX = [(find_features(rev), category) for (rev, category) in document_train]
testX = [(find_features(rev), category) for (rev, category) in document_test]

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(trainX)
print "\nSVC_classifier Test accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testX))*100
