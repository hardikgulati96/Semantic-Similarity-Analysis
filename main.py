from __future__ import division
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys
import path_len_sim
import m2
import tfidf
import pandas as pd 
import random
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter 
import plotly.plotly as py
style.use('fivethirtyeight')

str1 = "I was given a card by her in the garden."
str2 = "hero was great"

print("one",tfidf.cosine_sim(str1,str2))
print("two",m2.similarity(str1,str2))
print("three",path_len_sim.similarity(str1,str2,False))
print("four",path_len_sim.similarity(str1,str2,True))

p1=tfidf.cosine_sim(str1,str2)
p2=m2.similarity(str1,str2)
p3=path_len_sim.similarity(str1,str2,False)
p4=path_len_sim.similarity(str1,str2,True)





objects = ('Tfidf', 'Lsi', 'Dist-f', 'Dist-t')
y_pos = np.arange(len(objects))
performance = [p1,p2,p3,p4]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Similarity')
plt.title('Analysis of Semantic Similarity')
plt.savefig('Analysis.jpg')
plt.show()



