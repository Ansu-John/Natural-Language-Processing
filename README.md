### OBJECTIVE
Natural Language Processing or NLP is a field of Artificial Intelligence that gives the machines the ability to read, understand and derive meaning from human languages. This PoC was done to explore the different NLP options using Python. 

### DATASET USED
Dataset is taken from UCI Repository [SMS Spam Detection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

### TOOLS
Spark, Python - PySpark, NaiveBayes, Pipeline, MulticlassClassificationEvaluator and Text manipulations using PySpark

# Natural Language Processing 
Natural language processing (NLP) is an exciting field in data science and artificial intelligence that deals with teaching computers how to extract meaning from text. In this guide, we’ll be touring the essential stack of Python NLP libraries.

These packages handle a wide range of tasks such as part-of-speech (POS) tagging, sentiment analysis, document classification, topic modeling, and much more.

## NLTK 
NLTK is the most famous Python NLP library, and it’s led to incredible breakthroughs in the field. NLTK is responsible for conquering many text analysis problems.NLTK is popular for education and research. It’s heavy and slippery, and it has a steep learning curve. The second major weakness is that it’s slow and not production-ready.
Its modularized structure makes it excellent for learning and exploring NLP concepts, but it’s not meant for production.

## TextBlob 
TextBlob sits on the mighty shoulders of NLTK and another package called Pattern. This is our favorite library for fast-prototyping or building applications that don’t require highly optimized performance.

TextBlob makes text processing simple by providing an intuitive interface to NLTK. It’s a welcome addition to an already solid lineup of Python NLP libraries because it has a gentle learning curve while boasting a surprising amount of functionality.

By default, the sentiment analyzer is the PatternAnalyzer from the Pattern library. But what if you wanted to use a Naive Bayes analyzer? You can easily swap to a pre-trained implementation from the NLTK library.

```sh
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
opinion = TextBlob("EliteDataScience.com is dope!", analyzer=NaiveBayesAnalyzer())
opinion.sentiment
```
## CoreNLP
Stanford’s CoreNLP is a Java library with Python wrappers. It’s in many existing production systems due to its speed.
## SpaCy 
SpaCy is a new NLP library that’s designed to be fast, streamlined, and production-ready. It’s not as widely adopted, but if you’re building a new application, you should give it a try.

## Gensim 
Gensim is most commonly used for topic modeling and similarity detection. It’s not a general-purpose NLP library, but for the tasks it does handle, it does them well.

