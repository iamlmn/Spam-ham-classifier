import pandas as pd
import nltk
from nltk.corpus import stopwords

#messages=pd.read_csv('spamnewtxt.csv',encoding='latin-1')

messages = [line.rstrip() for line in open('spamnewtxt.txt')]
print(len(messages))
for message_no, message in enumerate(messages[:10]):
    print( message_no)
    print(message)


messages=pd.read_csv('spamnewtxt.txt',sep='\t',encoding='latin-1',names=['label','message'])
print(messages.head())
print(messages.describe())
print(messages.groupby('label').describe())

messages['length'] = messages['message'].apply(len)
print(messages.head())

import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
messages['length'].plot(bins=50, kind='hist').figure.savefig('dsad')
#messages['length'].plot(bins=50, kind='hist').savefig('dsad')

print(messages.length.describe())

print(messages[messages['length'] == 910]['message'].iloc[0])


x=messages.hist(column='length', by='label', bins=50,figsize=(10,4))
for i in x:
    i.figure.savefig('plotbylabel-eda')
import string
mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]
print(nopunc)
# Join the characters again to form the string.
nopunc = ''.join(nopunc)

print(nopunc)


from nltk.corpus import stopwords
stopwords.words('english')[0:10]

nopunc.split()

clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


print(messages['message'].head(5).apply(text_process))


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print (len(bow_transformer.vocabulary_))
message4 = messages['message'][3]
print (message4)
bow4 = bow_transformer.transform([message4])
print (bow4)
print (bow4.shape)
messages_bow = bow_transformer.transform(messages['message'])

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
#Given TF-IDF

messages_tfidf = tfidf_transformer.transform(messages_bow)


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])




all_predictions  = spam_detect_model.predict(messages_tfidf)
print (all_predictions)

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))

#Lets try it with train and test data cehck precision,recall and f1 score

from sklearn.cross_validation import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)

#pipeling

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)

print( classification_report(predictions,label_test))
