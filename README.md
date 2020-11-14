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

# Text manipulation using PySpark
PySpark provides various classes for text manipulation. They are roughly divided into these groups:

## Extraction: Extracting features from “raw” data
 - TF-IDF: 
Term frequency-inverse document frequency (TF-IDF) is a feature vectorization method widely used in text mining to reflect the importance of a term to a document in the corpus. 

 - Word2Vec: 
Word2Vec is an Estimator which takes sequences of words representing documents and trains a Word2VecModel. The model maps each word to a unique fixed-size vector. The Word2VecModel transforms each document into a vector using the average of all words in the document; this vector can then be used as features for prediction, document similarity calculations, etc. 

 - CountVectorizer
CountVectorizer and CountVectorizerModel aim to help convert a collection of text documents to vectors of token counts. When an a-priori dictionary is not available, CountVectorizer can be used as an Estimator to extract the vocabulary, and generates a CountVectorizerModel. The model produces sparse representations for the documents over the vocabulary, which can then be passed to other algorithms like LDA.During the fitting process, CountVectorizer will select the top vocabSize words ordered by term frequency across the corpus. 
This is especially useful for discrete probabilistic models that model binary, rather than integer, counts.

 - FeatureHasher
Feature hashing projects a set of categorical or numerical features into a feature vector of specified dimension (typically substantially smaller than that of the original feature space). This is done using the hashing trick to map features to indices in the feature vector.
The FeatureHasher transformer operates on multiple columns. Each column may contain either numeric or categorical features. Null (missing) values are ignored (implicitly zero in the resulting feature vector). Behavior and handling of column data types is as follows:

	- Numeric columns: For numeric features, the hash value of the column name is used to map the feature value to its index in the feature vector. By default, numeric features are not treated as categorical (even when they are integers). To treat them as categorical, specify the relevant columns using the categoricalCols parameter.
	
   -  String columns: For categorical features, the hash value of the string “column_name=value” is used to map to the vector index, with an indicator value of 1.0. Thus, categorical features are “one-hot” encoded (similarly to using OneHotEncoder with dropLast=false).
   
   -  Boolean columns: Boolean values are treated in the same way as string columns. That is, boolean features are represented as    “column_name=true” or “column_name=false”, with an indicator value of    1.0.
   
## Transformation: Scaling, converting, or modifying features
- Tokenizer
Tokenization class provides this functionality of taking text (such as a sentence) and breaking it into individual terms (usually words). RegexTokenizer allows more advanced tokenization based on regular expression (regex) matching. By default, the parameter “pattern” (regex, default: "\\s+") is used as delimiters to split the input text. Alternatively, users can set parameter “gaps” to false indicating the regex “pattern” denotes “tokens” rather than splitting gaps, and find all matching occurrences as the tokenization result.

- StopWordsRemover
Stop words are words which should be excluded from the input, typically because the words appear frequently and don’t carry as much meaning.
StopWordsRemover takes as input a sequence of strings (e.g. the output of a Tokenizer) and drops all the stop words from the input sequences. The list of stopwords is specified by the stopWords parameter. Default stop words for some languages are accessible by calling StopWordsRemover.

- n-gram
An n-gram is a sequence of n tokens (typically words) for some integer n. The NGram class can be used to transform input features into n-grams.
NGram takes as input a sequence of strings (e.g. the output of a Tokenizer). The parameter n is used to determine the number of terms in each n-gram. The output will consist of a sequence of n-grams where each n-gram is represented by a space-delimited string of n consecutive words. If the input sequence contains fewer than n strings, no output is produced.

- PCA
PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. A PCA class trains a model to project vectors to a low-dimensional space using PCA. 

- PolynomialExpansion
Polynomial expansion is the process of expanding your features into a polynomial space, which is formulated by an n-degree combination of original dimensions. A PolynomialExpansion class provides this functionality. The example below shows how to expand your features into a 3-degree polynomial space.

- StringIndexer
StringIndexer encodes a string column of labels to a column of label indices. 

- IndexToString
Symmetrically to StringIndexer, IndexToString maps a column of label indices back to a column containing the original labels as strings. 

- OneHotEncoder
One-hot encoding maps a categorical feature, represented as a label index, to a binary vector with at most a single one-value indicating the presence of a specific feature value from among the set of all feature values. This encoding allows algorithms which expect continuous features, such as Logistic Regression, to use categorical features. For string type input data, it is common to encode categorical features using StringIndexer first.

- VectorIndexer
VectorIndexer helps index categorical features in datasets of Vectors. It can both automatically decide which features are categorical and convert original values to category indices.

- Normalizer
Normalizer is a Transformer which transforms a dataset of Vector rows, normalizing each Vector to have unit norm. It takes parameter p, which specifies the p-norm used for normalization. (p=2 by default.) This normalization can help standardize your input data and improve the behavior of learning algorithms.

- StandardScaler
StandardScaler transforms a dataset of Vector rows, normalizing each feature to have unit standard deviation and/or zero mean. It takes parameters:
	- withStd: True by default. Scales the data to unit standard deviation.
	- withMean: False by default. Centers the data with mean before scaling. It will build a dense output, so take care when applying to sparse input.
	
- VectorAssembler
VectorAssembler is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees.

## Selection: Selecting a subset from a larger set of features
- VectorSlicer
VectorSlicer is a transformer that takes a feature vector and outputs a new feature vector with a sub-array of the original features. It is useful for extracting features from a vector column. VectorSlicer accepts a vector column with specified indices, then outputs a new vector column whose values are selected via those indices. There are two types of indices,
	- Integer indices that represent the indices into the vector, setIndices().
	- String indices that represent the names of features into the vector, setNames(). This requires the vector column to have an AttributeGroup since the implementation matches on the name field of an Attribute.
	- 
- RFormula
RFormula selects columns specified by an R model formula. Currently we support a limited subset of the R operators, including ‘~’, ‘.’, ‘:’, ‘+’, and ‘-‘. The basic operators are:
	- ~ separate target and terms
	- + concat terms, “+ 0” means removing intercept
	- - remove a term, “- 1” means removing intercept
	- : interaction (multiplication for numeric values, or binarized categorical values)
	- . all columns except target
RFormula produces a vector column of features and a double or string column of label.

- ChiSqSelector
ChiSqSelector stands for Chi-Squared feature selection. It operates on labeled data with categorical features. ChiSqSelector uses the Chi-Squared test of independence to decide which features to choose. It supports five selection methods: numTopFeatures, percentile, fpr, fdr, fwe:

	- numTopFeatures chooses a fixed number of top features according to a chi-squared test. This is akin to yielding the features with the most predictive power.
	- percentile is similar to numTopFeatures but chooses a fraction of all features instead of a fixed number.
	- fpr chooses all features whose p-values are below a threshold, thus controlling the false positive rate of selection.
	- fdr uses the Benjamini-Hochberg procedure to choose all features whose false discovery rate is below a threshold.
	- fwe chooses all features whose p-values are below a threshold. The threshold is scaled by 1/numFeatures, thus controlling the family-wise error rate of selection. By default, the selection method is numTopFeatures, with the default number of top features set to 50. The user can choose a selection method using setSelectorType.
