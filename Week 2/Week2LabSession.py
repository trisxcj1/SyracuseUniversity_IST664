import nltk
from nltk import FreqDist

nltk.download('gutenberg')
nltk.download('punkt')

# get the text of the book Emma from the Gutenberg corpus, tokenize it,
#   and reduce the tokens to lowercase.
file0 = nltk.corpus.gutenberg.fileids( ) [0]
emmatext = nltk.corpus.gutenberg.raw(file0)
emmatokens = nltk.word_tokenize(emmatext) 
emmawords = [w.lower( ) for w in emmatokens] 
# show some of the words
print(len(emmawords))
print(emmawords[ :110])


# Creating a frequency distribution of words
ndist = FreqDist(emmawords)

# print the top 30 tokens by frequency
nitems = ndist.most_common(30)
for item in nitems:
    print (item[0], '\t', item[1])

    
# look at other tokenization from the corpus
emmawords2 = nltk.corpus.gutenberg.words('austen-emma.txt')
emmawords2lowercase = [w.lower() for w in emmawords2]

print(emmawords[:160])
print(emmawords2lowercase[:160])

import re

# this regular expression pattern matches any word that contains all non-alphabetical
#   lower-case characters [^a-z]+
# the beginning ^ and ending $ require the match to begin and end on a word boundary 
pattern = re.compile('^[^a-z]+$')

nonAlphaMatch = pattern.match('**')
#  if it matched, print a message
if nonAlphaMatch: print ('matched non-alphabetical')
  
# function that takes a word and returns true if it consists only
#   of non-alphabetic characters  (assumes import re)
def alpha_filter(w):
  # pattern to match word of non-alphabetical characters
  pattern = re.compile('^[^a-z]+$')
  if (pattern.match(w)):
    return True
  else:
    return False
 
# apply the function to emmawords
alphaemmawords = [w for w in emmawords if not alpha_filter(w)]
print(alphaemmawords[:100])
print(len(alphaemmawords))

# get a list of stopwords from nltk
nltkstopwords = nltk.corpus.stopwords.words('english')
print(len(nltkstopwords))
print(nltkstopwords)

# check tokenization in emmawords
print(emmawords[:100])
print(emmawords[15300:15310])

morestopwords = ['could','would','might','must','need','sha','wo','y',"'s","'d","'ll","'t","'m","'re","'ve", "n't"]
stopwords = nltkstopwords + morestopwords
print(len(stopwords))
print(stopwords)

stoppedemmawords = [w for w in alphaemmawords if not w in stopwords]
print(len(stoppedemmawords))

#  use this list for a better frequency distribution
emmadist = FreqDist(stoppedemmawords)
emmaitems = emmadist.most_common(30)
for item in emmaitems:
  print(item)
 

# Bigrams and mutual information
# Bigrams and Bigram frequency distribution
emmabigrams = list(nltk.bigrams(emmawords))
print(emmawords[:21])
print(emmabigrams[:20])

# setup for bigrams and bigram measures
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()

# create the bigram finder and score the bigrams by frequency
finder = BigramCollocationFinder.from_words(emmawords) # pass a complete list of words before any filtering (ie with punctuation etc)
scored = finder.score_ngrams(bigram_measures.raw_freq)

# scored is a list of bigram pairs with their score
print(type(scored))
first = scored[0]
print(type(first))
print(first)

# scores are sorted in decreasing frequency
for bscore in scored[:30]:
    print (bscore)
    
# apply a filter to remove non-alphabetical tokens from the emma bigram finder
finder.apply_word_filter(alpha_filter)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:30]:
    print (bscore)
    
# apply a filter to remove stop words
finder.apply_word_filter(lambda w: w in stopwords)
scored = finder.score_ngrams(bigram_measures.raw_freq)
for bscore in scored[:20]:
    print (bscore)
    
### pointwise mutual information
finder3 = BigramCollocationFinder.from_words(emmawords)
scored = finder3.score_ngrams(bigram_measures.pmi)
for bscore in scored[:20]:
    print (bscore)
    
# to get good results, must first apply frequency filter
finder.apply_freq_filter(5)
scored = finder.score_ngrams(bigram_measures.pmi)
for bscore in scored[:30]:
    print (bscore)
