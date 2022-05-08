import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
import re
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")

url = 'https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv'

# Extracting the Raw Data from the github url [unredactor.tsv]
def Readdata(url):
    df = pd.read_csv(url, sep="\t", on_bad_lines='skip', quoting=csv.QUOTE_NONE)
    df = df.rename(columns={'cegme': 'usernames', 'training': 'dtype', 'ashton kutcher': 'names', "I couldn't image ██████████████ in a serious role, but his performance truly": 'sentences'})
    df.isnull().sum()
    df.dropna(inplace=True)
    df.isnull().sum()
    return df

# Preprocessing the data by removing the stopwords
def preprocess(df):
    cleanlist=[]
    k=[]
    stop_words=set(stopwords.words("english"))
    for i in df['sentences']:
        i = "".join(i)
        k.append(i)
    #print(k)

    for j in k:
        j = j.lower()
        j = ''.join('' if j.isdigit() else c for c in j)
        token = nltk.word_tokenize(j)
        mytokens = " ".join([word for word in token if word not in stop_words])
        cleanlist.append(mytokens)
    df['sentences']=cleanlist
    return df

# Finding the Sentiment score
def Sentimentscore(df):
    positivity_list = []
    SIA = SentimentIntensityAnalyzer()
    for i in range(len(df)):
        sent=SIA.polarity_scores(df['sentences'].iloc[i])
        positivity_list.append(sent['pos'])
    df7 = pd.DataFrame(positivity_list)
    df7.columns = ['Poitivity']
    combi = [df, df7]
    df = pd.concat(combi, axis=1)
    return df

# Finding the n_grams of the readcted sentences
def fngrams(df):
    ngram=4
    n_grams = []
    for text in df['sentences']:
        words=[word for word in text.split(" ")]
        temp=zip(*[words[i:] for i in range(0,ngram)])
        ans=[' '.join(ngram) for ngram in temp]
        n_grams.append(ans)
    #print(n_grams)
    df['ngrams']=n_grams
    return df

# Vectorizing the sentences and storing into the dataframe for further usage
def Vectorize(df):
    vectorizer = CountVectorizer()
    #vector = vectorizer.fit_transform(df['sentences'])
    vector = vectorizer.fit_transform(df['sentences'])
    df5 = pd.DataFrame(vector.toarray())
    comb = [df, df5]
    df = pd.concat(comb, axis=1)
    df3 = df.drop(labels = ['usernames', 'sentences', 'ngrams'], axis = 1)
    return df3

# Training the model and predicting the Precision, Recall and F-1 Scores using the test data
def Prediction(df3):
    df4 = df3[(df3['dtype']=='training') | (df3['dtype']=='validation')]
    df5 = df3[df3['dtype']=='testing']
    x=df4.loc[:,'Poitivity':]
    y=df4['names']
    z=df5.loc[:,'Poitivity':]
    n=df5['names']
    SVM = svm.LinearSVC(max_iter=5000)
    clf = SVM.fit(x,y)
    y_predi = SVM.predict(z)
    print("Model Accuracy :",accuracy_score(n, y_predi) * 100)
    print("Precision:",metrics.precision_score(n,y_predi,average='weighted', zero_division='warn'))
    print("Recall:",metrics.recall_score(n,y_predi,average='weighted'))
    print("F-1 Score:",metrics.f1_score(n,y_predi,average='weighted'))

a=Readdata(url)
b=preprocess(a)
c=Sentimentscore(b)
d=fngrams(c)
e=Vectorize(d)
Prediction(e)
