#         multi document summarization using pmi and cosine similarity and wordnet words changing

#group:zain ul abdeen(cs141058)
#haseef sachwani(cs141045)

from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

import scipy.spatial as sp
from nltk.corpus import stopwords
from string import punctuation
from bs4 import BeautifulSoup
import codecs
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from math import exp, expm1 ,log,ceil
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine


def gettext(f): #getting text from dawn article
    # kill all script and style elements
    soup = BeautifulSoup(f,'html.parser')
    for script in soup("script", "style"):
        script.extract()    # rip it out
    text= ""
    # get text
    for row in soup.find_all('p',attrs={"class" : r""}):
       text+=row.text

    text = text.replace("Dear reader, please upgrade to the latest version of IE to have a better reading experience","")
    text = text.replace("Copyright © 2017Scribe Publishing Platform","")
    
    abc="""Dear readers,As we greatly value the trust you place in us, we think it fit to apprise you that over the last three months, the
    Dawn Media Group has been under cyber attack. Multiple attempts have been made to hack and hijack our
    official social media accounts and the accounts of our staff.If such attempts were to succeed, the Dawn Media Group would lose control over what gets published and the
    hacker(s) would acquire the dreadful power to post false, harmful and dangerous content to give an impression
    to the reader that the offensive posts emanated from Dawn or with its acquiescence. Hackers and malicious
    groups attempt to damage reputations and stir up virulent and negative campaigns to serve their agenda, and
    Dawn has already been a target of such online campaigns in the past.While we are continuing to fortify our digital security, you may draw some comfort knowing that Dawn would
    never, under any conceivable circumstance, post or condone the posting of content, either on any website or its
    social media accounts, that is in violation of the law and our code of ethics, particularly hate speech, and content
    that is blasphemous, indecent, prohibited, illegal, or likely to arouse societal discord or disturb tranquility.We hope you will stand with us in the fight against hackers who are bent upon spreading fear, hatred and disin-formation;
    who through their actions, are attacking our shared commitment to the progress of Pakistan."""
    text = text.replace(abc,"")
        
    
    return text

def preprocess(text): #from tokenizing in setence,then stopwords removal then stemming
    sentence = sent_tokenize(text)

    words = word_tokenize(text)

    #lower case
    s=0
    for w in sentence:
       sentence[s] = w.lower()
       s=s+1
    fw = 0
    for wo in words:
      words[fw] = wo.lower()
      fw=fw+1

    stopword = set(stopwords.words('english') + list(punctuation))

    text_stop = []

    #stop words removal
    for w in words:
        if w not in (stopword and text_stop):
            text_stop.append(w)

    #bagsofwords = [collections.Counter(words for words in text_stop) for line in sentence]


    ps=PorterStemmer()
    pc = 0
    for w in text_stop:
      text_stop[pc]=ps.stem(w)
      pc = pc+1
  
    vocab = []
    for w in text_stop:
        if w not in vocab:
            vocab.append(w)

    return vocab,sentence

def bow(vocab,sentence):
    count_vect = CountVectorizer(vocabulary=vocab)

    bagofwords = count_vect.transform(sentence)
    bagsofwords = bagofwords.toarray()
    bagsofwords=bagsofwords.transpose()
    
    return bagsofwords

def sumofmatrix(bagsofwords):
    sum = 0
    for i in bagsofwords:
        for j in i:
            sum=sum+j
            
    return sum

def pwfunc(bagsofwords,sum):
    pw = []
    for i in bagsofwords:
        for j in i:
            temp = j/sum
            pw.append(temp)
    return pw
    
def psfunc(bagsofwords,sum):
    temp3 = 0.0
    ps = []
    sumofcoloums = bagsofwords.sum(axis=0)
    for i in sumofcoloums:
        temp3 = i/sum
        ps.append(temp3)
    return ps

def pwifunc(bagsofwords,sum):
    temp2 = 0.0
    pwi = []
    sumsofrows = bagsofwords.sum(axis=1)

    for i in sumsofrows:
        temp2 = i/sum
        pwi.append(temp2)
    return pwi

def pmi(sum,bagsofwords,vocab,sentence,pw,pwi,ps): #calculates pmi value fow word,sen matrix
    
    temp = 0.0
    ppmi = []

    temp1 = 0.0
    pwc = 0
    pwic = -1
    psc = 0

    a=0.0
    b=0.0
    c=0.0
    d=0.0
    bagsofwords1 = bagsofwords.astype(float)

    for i in range(vocab):
        pwic = pwic+1
        for j in range(sentence):
            if(pw[pwc] != 0.0):
                a = log(pw[pwc],10)
            
            if(ps[psc] != 0.0):
                c = log(ps[psc],10)
            
            if(pwi[pwic] != 0.0):
                b = log(pwi[pwic],10)

            d=b+c
            temp1 = a-d
            bagsofwords1[i][j] = temp1
            pwc=pwc+1
            psc=psc+1
            if((psc) >= len(ps)):
                psc = 0
    return bagsofwords1

def calcweight(bagsofwords1,ps):
    temp=0.0
    weights=[]
    sumofcoloums = bagsofwords1.sum(axis=0)
    for i in range(len(ps)):
        temp=sumofcoloums[i]*ps[i]
        weights.append(temp)
    return weights




def tag(sentence):
  words = word_tokenize(sentence)
  words = pos_tag(words)
  return words

def paraphraseable(tag):
  return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')

def pos(tag):
  if tag.startswith('NN'):
    return wn.NOUN
  elif tag.startswith('V'):
    return wn.VERB

def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word, pos(tag))]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)

def synonymIfExists(sentence):
  for (word, t) in tag(sentence):
    if paraphraseable(t):
      syns = synonyms(word, t)
      if syns:
        if len(syns) > 1:
          yield [word, list(syns)]
          continue
    yield [word, []]

def paraphrase(sentence):
  return [x for x in synonymIfExists(sentence)]



    

    
   

    

    
    
   
    
    
    
 #main function   


if __name__ == '__main__':
    '''document 1 summary'''
    weights = []
    pwi = []
    ps = []
    pw = []
    sum1 = 0
    #f = codecs.open(r"C:\Users\zain_\Desktop\v.html",'r', 'utf-8')
    f = codecs.open(r"kholi1.html",'r', 'utf-8')
    text = gettext(f)
    vocab,sentence = preprocess(text)
    bagsofwords = bow(vocab,sentence)
    sum1=sumofmatrix(bagsofwords)
    pw=pwfunc(bagsofwords,sum1)
    ps=psfunc(bagsofwords,sum1)
    pwi=pwifunc(bagsofwords,sum1)
    
    bagsofwords1 = pmi(sum1,bagsofwords,len(vocab),len(sentence),pw,pwi,ps)

    weights=calcweight(bagsofwords1,ps)

    temp = 0.0
    temp1 = ""
    for i in range(len(weights)):
       for j in range(len(weights)):   
           if(weights[i]>=weights[j]):
               temp=weights[i]
               weights[i]=weights[j]
               weights[j]=temp
               temp1=sentence[i]
               sentence[i]=sentence[j]
               sentence[j]=temp1

    
    percent = 5
    percentage = ceil((len(sentence)*percent)/100)
    #print(percentage)
    #print([sentence[i] for i in range(percentage)] )
    print(percentage)
    print(bagsofwords.shape)
    '''for i in range(percentage):
        print(str(sentence[i])+"\r")'''

    '''document 2 summary'''

 
    weights1 = []
    pwi1 = []
    ps1 = []
    pw1 = []
    sum2 = 0
    #f = codecs.open(r"C:\Users\zain_\Desktop\v.html",'r', 'utf-8')
    f1 = codecs.open(r"kholi3.html",'r', 'utf-8')
    text1 = gettext(f1)
    vocab1,sentence1 = preprocess(text1)
    bagsofwords2 = bow(vocab1,sentence1)
    sum2=sumofmatrix(bagsofwords2)
    pw1=pwfunc(bagsofwords2,sum2)
    ps1=psfunc(bagsofwords2,sum2)
    pwi1=pwifunc(bagsofwords2,sum2)
    
    bagsofwords3 = pmi(sum2,bagsofwords2,len(vocab1),len(sentence1),pw1,pwi1,ps1)

    weights1=calcweight(bagsofwords3,ps1)

    temp = 0.0
    temp1 = ""
    for i in range(len(weights1)):
       for j in range(len(weights1)):   
           if(weights1[i]>=weights1[j]):
               temp=weights1[i]
               weights1[i]=weights1[j]
               weights1[j]=temp
               temp1=sentence1[i]
               sentence1[i]=sentence1[j]
               sentence1[j]=temp1

    
    percent1 = 5
    percentage1 = ceil((len(sentence1)*percent1)/100)
    #print(percentage)
    #print([sentence[i] for i in range(percentage)] )
    print(percentage1)
    print(bagsofwords2.shape)
    '''for i in range(percentage):
        print(str(sentence1[i])+"\r")'''

    '''multidoc'''

    summary=[]
    summary1=[]
    for i in range(percentage):
        summary.append(sentence[i])

    for i in range(percentage1):
        summary1.append(sentence1[i])


    text3=''
    text4=''

    '''for i in summary:
        text3=text3+i'''
    for i in range(percentage):
        text3=text3+sentence[i]
        
    for i in range(percentage1):
        text4=text4+sentence1[i]
        
    '''g = codecs.open('file2.txt','w')
    g.write(text3)'''
    vocab2,sentence2 = preprocess(text3)
    bofw = bow(vocab2,summary)
    vocab3,sentence3 = preprocess(text4)
    fvocab=[]
    for i in vocab2:
        fvocab.append(i)

    for i in vocab3:
        if(i not in fvocab):
            fvocab.append(i)
    bofw = bow(fvocab,summary)
    bofw=bofw.transpose()
    bofw1 = bow(fvocab,summary1)
    bofw1=bofw1.transpose()
    
    
   
    #dist_out = 1-pairwise_distances(bofw, metric="cosine")
    print(bofw.shape)
    print(bofw1.shape)
    bofw2= 1 - sp.distance.cdist(bofw1, bofw, 'cosine')
    print(bofw2)
    count=-1
    count1=-1
    fsummary=[]
    for i in bofw2:
        count1=-1
        count+=1
        for j in i:
            count1+=1
            if(j >=0.6):
                if(summary1[count] not in fsummary):
                    fsummary.append(summary1[count])
                if(summary[count1] not in fsummary):
                    fsummary.append(summary[count1])
    #for i in fsummary:
     #   print(i)

    g=codecs.open('file2.txt','w')

    for i in fsummary:
        g.write(i+"\r")


        #wordnet similar words are changed here
    list1=[]

    
    for i in fsummary:
        list1.append(paraphrase(i))
    etc= ""
    fsummary5=[]
    for i in list1:
        for j in i:
            listz=j[1]
            if(len(listz)>=1):
                etc=etc+listz[0]
                etc=etc+" "
            else:
                etc=etc + j[0]
                etc=etc+" "
        fsummary5.append(etc)
        etc=""
    

    h = codecs.open('file3.txt','w')

    for f in fsummary5:
        h.write(f +"\r")

    print("multi document summary in file 2.txt"+ "\n"+ "multi document summary with wordsnet synonyms replaced in file 3.txt")
