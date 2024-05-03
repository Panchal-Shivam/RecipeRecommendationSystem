#!/usr/bin/env python
# coding: utf-8

# # Recipe Recommendation System
# Creating, Testing, and Tuning unsupervised learning methods to recommend relevant recipes based on ingredient and category preference
# 
# Workflow:
# 1. Load, aggregate, clean, and tokenize recipe text data
#     - Identify Food Recipe Specific stop words that might be useful to ignore (e.g. measurements, numbers)
#     - It's likely that recipe attributes will depend heavily on ingredients and cooking methods, not necessarily
#     - Think twice about assigning words as stopwords, they might end up being useful.
#     - You may want to lemmatize the data to reduce sparseness; check lemmatization process to see if it strips important words, foods, or ingredients. Also, remove punctuation: it won't help for keyword 
# 2. ~~Create Word Embeddings using Word2Vec or GloVe Models (Consider using pretrained word embeddings)~~
#     - Discuss in detail the reason to choose one over the other for this context
#     - Setup neural network locally and run remotely on google colab
#     - I want to try tfidf and GloVe model, because tfidf doesn't take into account the order of words, which is isn't such a problem with recipes - it's the ingredients and cooking techniques that matter more. However, GloVe word vectors may be able to produce new words outside of the corpora text when summarizing the documents
# 3. Compare Topic Extraction Methods
#     - ~~LDA2Vec~~
#     - LDA
#     - NNMF
# 4. Generate keywords using keyword summarization and textrank
#     - Create methodology selectively assigns generated categories (e.g. LSA/NNMF Score must be above certain score threshold)
#     - Define metrics that evaluate the validity, breadth, and descriptive value of the assigned categories
#     - Identify Food Recipe Specific stop words that passed that might be suitable in the filter.
# 5. Create extra features useful for search result ranking
#     - ~~Difficulty (Time, Number of Ingredients, Servings (inverse relationship),~~
#     - Import Unsupervised Generated Categories
#     - Create ratings that calculate overall weight.
# 6. Find similarity scoring methods that would work best in this context. Some variation of Cosine Similarity will work best
# 7. Create algorithm that utilizes similarity to sort recipes based on user-inputted queries, and sort base on other features as well.
# 
# -----
# 
# Regarding the Available Recipe Images
# 
# Around 70,000 recipes out of the 125,000 have corresponding images, so it's possible to utilize these images to improve the models or create seperate, supplementary model
# 
# Ideas:
# - Training a neural network to identify/predict/generate categories of foods based on their images
# 

# ### The Data
# Although I can no longer find the direct download, the link and code for the scraper the original user used to collect the data set is [here](https://github.com/rtlee9/recipe-box). This user collected the title, ingredients, and instructions from recipes found on Allrecipes.com, epicurious.com, and foodnetwork.com.
# 
# For this project, the data was directly downloaded and uploaded for the creation of my own model. All the code present are of my own creation or significantly modified from the Thinkful curriculum. No other software sources were used verbatim within this project.

# In[1]:


import pandas as pd
import numpy as np
import re
import spacy


# In[18]:


allrecipes_raw = pd.read_json('../__DATA__/recipes_raw/recipes_raw_nosource_ar.json')
epicurious_raw = pd.read_json('../__DATA__/recipes_raw/recipes_raw_nosource_epi.json')
foodnetwork_raw = pd.read_json('../__DATA__/recipes_raw/recipes_raw_nosource_fn.json')


# In[132]:


allrecipes = allrecipes_raw.copy().T.reset_index().drop(columns = ['index'])
allrecipes.head(1)


# In[133]:


epicurious = epicurious_raw.copy().T.reset_index().drop(columns = ['index'])
epicurious.head(1)


# In[134]:


foodnetwork = foodnetwork_raw.copy().T.reset_index().drop(columns = ['index'])
foodnetwork.head(1)


# In[135]:


recipes = pd.concat([allrecipes, epicurious, foodnetwork]).reset_index(drop=True) # Concat does not reset indices
recipes.shape


# In[136]:


# Count of missing values by category
recipes.isna().sum()


# In[137]:


# Number recipes/rows that have any missing values besides missing pictures
null_recs = recipes.copy().drop(columns = 'picture_link').T.isna().any()
null_recs.sum()


# In[138]:


recipes[null_recs].head()


# In[139]:


rows_to_drop = recipes[null_recs].index
recipes = recipes.drop(index = rows_to_drop).reset_index(drop = True)
recipes.shape


# In[140]:


recipes.dtypes


# In[141]:


# Indexing rows with columns that only contain numbers or punctuation
import string
nc_ingred_index = [index for i, index in zip(recipes['ingredients'], recipes.index) if all(j.isdigit() or j in string.punctuation for j in i)]
nc_title_index = [index for i, index in zip(recipes['title'], recipes.index) if all(j.isdigit() or j in string.punctuation for j in i)]
nc_instr_index = [index for i, index in zip(recipes['instructions'], recipes.index) if all(j.isdigit() or j in string.punctuation for j in i)]


# In[142]:


# Checking number of rows in each category that are only punc/nums
index_list = [nc_ingred_index, nc_title_index, nc_instr_index]
[len(x) for x in index_list]


# In[143]:


# generating unique indices for index_list and dropping from dataframe
# recipes without recipe instructions or ingredients are not useable
from functools import reduce
from operator import add
inds_to_drop = set(reduce(add, index_list))
print(len(inds_to_drop))
recipes = recipes.drop(index=inds_to_drop).reset_index(drop=True)
recipes.shape


# In[146]:


# Recipe instructions with less than 20 characters are not good recipes
empty_instr_ind = [index for i, index in zip(recipes['instructions'], recipes.index) if len(i) < 20]
recipes = recipes.drop(index = empty_instr_ind).reset_index(drop=True)


# In[147]:


recipes.shape


# In[148]:


recipes.isna().sum()


# In[159]:


# Checking for low ingredient recipes.
#low_ingr_ind = [index for i, index in zip(recipes['ingredients'], recipes.index) if len(i) < 20]
low_ingr_index = [index for i, index in zip(recipes['ingredients'], recipes.index) if i[0] == np.nan]
len(low_ingr_index)
recipes.loc[low_ingr_index, 'ingredients']


# In[155]:


# Searching for pseudo empty lists
[index for i, index in zip(recipes['ingredients'], recipes.index) if np.nan in recipes.loc[index,'ingredients']]


# ### Cleaning to Prepare for Tokenizing
# 
# Cleaning Specifics:
# - Removing ADVERTISEMENT
# - Pruning dataset of rows with empty cells or inadequate recipes
# - Remove all punctuation, digits, and extraneous spacing

# In[196]:


# Removing ADVERTISEMENT text from ingredients list
ingredients = []
for ing_list in recipes['ingredients']:
    clean_ings = [ing.replace('ADVERTISEMENT','').strip() for ing in ing_list]
    if '' in clean_ings:
        clean_ings.remove('')
    ingredients.append(clean_ings)
recipes['ingredients'] = ingredients


# In[199]:


recipes.loc[0,'ingredients']


# In[200]:


# Extracting ingredients from their lists and formatting as single strings
recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in recipes['ingredients']]
recipes['ingredient_text'].head()


# In[201]:


# Counting the number of ingredients used in each recipe
recipes['ingredient_count'] = [len(ingredients) for ingredients in recipes['ingredients']]


# In[212]:


recipes.head(1)


# In[211]:


all_text = recipes['title'] + ' ' + recipes['ingredient_text'] + ' ' + recipes['instructions']
all_text[0]


# In[ ]:


# Clean_text Function
import string
import re

def clean_text(documents):
    cleaned_text = []
    for doc in documents:
        doc = doc.translate(str.maketrans('', '', string.punctuation)) # Remove Punctuation
        doc = re.sub(r'\d+', '', doc) # Remove Digits
        doc = doc.replace('\n',' ') # Remove New Lines
        doc = doc.strip() # Remove Leading White Space
        doc = re.sub(' +', ' ', doc) # Remove multiple white spaces
        cleaned_text.append(doc)
    return cleaned_text

# Cleaning Text
cleaned_text = clean_text(all_text)


# In[228]:


cleaned_text[2]


# ### Tokenizing Using Spacy

# For this tokenization, we will lemmatize the words. This is will help create a denser word embeddings. However, no POS tagging, know entities, or noun_phrases will be parsed and added.

# In[230]:


# Testing Strategies and Code
nlp = spacy.load('en')

' '.join([token.lemma_ for token in nlp(cleaned_text[2]) if not token.is_stop])


# My current strategy is to strip down the text as much as possible. In this case that means lemmatizing words and removing stop words. The goal here is not text prediction, but similarity measures and keyword extraction, which don't require the semantic granularity that stop words and non-lemmatized words might provide.

# In[231]:


# Tokenizing Function that lemmatizes words and removes Stop Words
def text_tokenizer(documents):
    tokenized_documents = []
    for doc in documents:
        tok_doc = ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])
        tokenized_documents.append(tok_doc)
    return tokenized_documents


# In[237]:


# Tokenizing Function to run in parallel
def text_tokenizer_mp(doc):
    tok_doc = ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])
    return tok_doc


# In[236]:


import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())


# In[242]:


# Parallelzing tokenizing process
pool = mp.Pool(mp.cpu_count())
tokenized_text = pool.map(text_tokenizer_mp, [doc for doc in cleaned_text])


# In[ ]:


# Save the tokenized_text variable as a csv in order to return to it;
# Do not attempt to run the parser above, it will simply take too long
# Reload the csv from file insted
pd.Series(tokenized_text).to_csv('tokenized_text.csv')


# In[244]:


tokenized_text[0]


# ### Creating Word Embeddings
# 
# - TF-IDF
# - Pre-trained GloVe Word Embeddings
# - GloVe Embeddings trained on the recipe corpora
# 
# In an attempt to create dense word embeddings, I could find no reliable examples to follow that integrate GloVe or Word2Vec with document topic modeling.

# In[251]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase = True,
                            ngram_range = (1,1))

text_tfidf = vectorizer.fit_transform(tokenized_text)
tfidf_words = vectorizer.get_feature_names()
print(text_tfidf.shape)
print(len(tfidf_words))


# ## Topic Modeling
# - LDA
# - NNMF
# 
# The ultimate goal with topic modeling is to group documents together and generate category words using TextRank. These category words can then be used to further refine the recommendation query
# 
# ------
# 
# LDA and NNMF extract topic models by finding similar subgroups of text within the corpora of recipes (or other text documents). However

# In[252]:


text_tfidf.shape


# In[253]:


from sklearn.decomposition import LatentDirichletAllocation as LDA

lda = LDA(n_components = 50,
          n_jobs = -1,
          max_iter = 100)
text_lda = lda.fit_transform(text_tfidf)
text_lda.shape


# In[254]:


from sklearn.decomposition import NMF

nmf = NMF(alpha=0.0,
         init='nndsvdar',
         l1_ratio=0.0,
         max_iter = 100,
         n_components = 50,
         solver='cd')

text_nmf = nmf.fit_transform(text_tfidf)
text_nmf.shape


# Models were arbitrarily set to 50 topics. Unfortunately, neither NNMF nor LDA have the ability to calculate the percentage of variance that they capture from the original tfidf matrix. So 50 topics is purely a shot in the dark.

# Next Steps:
# 1. Document x Topic Matrix
# 2. Word x Topic Matrix

# ## Exploring Topics by Document

# In[415]:


# variable dependencies:
text_series = pd.Series(all_text)

def docs_by_tops(top_mat, topic_range = (0,0), doc_range = (0,2)):
    for i in range(topic_range[0], topic_range[1]):
        topic_scores = pd.Series(top_mat[:,i])
        doc_index = topic_scores.sort_values(ascending = False)[doc_range[0]:doc_range[1]].index
        for j, index in enumerate(doc_index, doc_range[0]):
            print('Topic #{}'.format(i),
                  '\nDocument #{}'.format(j),
                  '\nTopic Score: {}\n\n'.format(topic_scores[index]),
                  text_series[index], '\n\n')


# In[402]:


docs_by_tops(text_lda,(0,3),(0,3))


# In[403]:


docs_by_tops(text_nmf,(0,3),(0,3))


# In[423]:


docs_by_tops(text_nmf,(1,2),(90000,90001))


# ### Exploring Topics by words

# In[302]:


text_nmf.shape


# In[304]:


text_tfidf.T.shape


# In[321]:


# Function for best topic words using cosine similarity
# Variable Dependency:
word_series = pd.Series(tfidf_words)

def words_by_tops(tfidf_mat, top_mat, topic_range=(0,0), n_words=10):
    topic_word_scores = tfidf_mat.T * top_mat
    for i in range(topic_range[0],topic_range[1]):
        word_scores = pd.Series(topic_word_scores[:,i])
        word_index = word_scores.sort_values(ascending = False)[:n_words].index
        print('\nTopic #{}'.format(i))
        for index in word_index:
            print(word_series[index],'\t\t', word_scores[index])


# In[322]:


# Keywords using LDA
words_by_tops(text_tfidf, text_lda, (0,3), 10)


# In[979]:


# Words using NMF
words_by_tops(text_tfidf, text_nmf, (0,3), 10)


# Ultimately, in looking at the first three topic documents for LDA and NNMF, it appears that NNMF made more distinct topic models: 0. Spreads, 1. Cakes 2. Chicken.
# 
# LDA on the other made two good topics 0. Salads and 1. Gravies. The third topic, I am unable to tell what it's clustering on. Therefore, we will proceed with NNMF topics to generate

# ### Keyword Extraction of Topics Using TextRank

# The purpose of using TextRank to extract keywords 
# 
# ------------
# 
# Consider using a smaller corpora size, so as to more quickly code. Then run the entire copora.
# Using the time module to test out corpora sizes.
# 
# Using TextRank to summarize the topics by extracting words involves many variables:
# - Deciding how many of the top documents each the topic should be summarized?
#     - Arbitrarily: top 100, evaluate, then only decrease from there.
# - Should TextRank then be performed once over the selected topic corpora, or should it be run individually and then scores added to make an aggregated rank?
#     - For the sake of simplicity it should probably only be run over the entire corpora
# - How many top ranked words should be used?
#     - Check out the top ranks words first, then decide.
#     - Arbitrarily choosing the top 20 ranked
# - Then once keywords are decided, to how many documents should those words be assigned the extract?
#     - I think it depends on the number of documents used to find the categorical keywords.
# - Once we're satisfied with a TextRank strategy, we need to this about which method to use to extract topics (NNMF, LDA) as well as whether we'll NNMF, LDA, or TextRank to extract keywords.

# In[584]:


# Pulling the top one-hundred documents ranked in similarity among Topic #1
text_index = pd.Series(text_nmf[:,1]).sort_values(ascending = False)[:100].index
text_4summary = pd.Series(cleaned_text)[text_index]

# Manually Creating a list of recipe stop
recipe_stopwords = ['cup','cups','ingredient','ingredients','teaspoon','tablespoon','oven']


# Because recipes are a very niche subject matter within NLP, it's likely that there are no list of stopwords related to this domain. The above 'recipe_stopwords' will thus be manually updated as necessary.

# In[585]:


# generating topic filter
import time
start_time = time.time()

parsed_texts = nlp(' '.join(text_4summary)) 
kw_filts = set([str(word) for word in parsed_texts 
                if (word.pos_== ('NOUN' or 'ADJ' or 'VERB'))
                and str(word) not in recipe_stopwords])

print('Execution Time: {} seconds', time.time() - start_time)


# In[586]:


# Creating adjecency Table for recipes.
adjacency = pd.DataFrame(columns=kw_filts, index=kw_filts, data = 0)
for i, word in enumerate(parsed_texts):
    if any ([str(word) == item for item in kw_filts]):
        end = min(len(parsed_texts), i+5) # Window of four words
        nextwords = parsed_texts[i+1:end]
        inset = [str(x) in kw_filts for x in nextwords]
        neighbors = [str(nextwords[i]) for i in range(len(nextwords)) if inset[i]]
        if neighbors:
            adjacency.loc[str(word), neighbors] += 1


# For the sake of implementing a simple TextRank precedure, all of the recipes were concatenated into one continuous corpora. The issue with this is that in tabulating the word adjacencies there is no delineation between one recipe and the next. So the neighbors at the tail ends of any one recipe are considered neighbors to the words at the front end of the next corresponding recipe. However, while this does generate false word adjacencies, the recipes are ordered by topic rank so that that adjacent recipes are in content. In theory, this dampens the effect, if any, that the false word adjacencies might create.

# In[574]:





# In[587]:


import networkx as nx

# Running TextRank
nx_words = nx.from_numpy_matrix(adjacency.values)
ranks=nx.pagerank(nx_words, alpha=.85, tol=.00000001)

# Identifying the most highly ranked keywords
ranked = sorted(((ranks[i],s) for i,s in enumerate(kw_filts)),
                reverse=True)


# In[591]:


ranked[:25]


# In[576]:


adjacency.shape


# In[578]:


# checking to see there are actual values loaded in the adjacency df
import scipy
scipy.sparse.csr_matrix(adjacency.copy().values)


# Looking just at the top ranked keywords for the Topic \#1 documents, one would guess that the topic that was clustered is baked cakes. The top ranked word is 'baking' which is spot on! Unfortunately, because we used tf-idf, there is no way of calculating from these keywords a sort of categorical vector mapping such as 'baking'. If we were to use some sort of word embeddings such as Word2Vec or GloVe then we might be able to calculate abstract summaries of these words.
# 
# However, these keyword extractions are still useful insofar that the queries will match indidrectly to the the umbrella topic of 'baking' if those queries still match with the words extracted from the baking topic.
# 
# For now, we will arbitratily choose between text rank and nnmf word topic ranks to generate keywords. TextRank is a clear winner over mere text cosine similarity ranks in the case of this topic, so we will proceed with using TextRank as the categories. 

# In[579]:


len(kw_filts)


# In[581]:


pd.Series(list(kw_filts)).nunique()


# In[397]:


text_4summary[3060]


# Exploration of TextRank Takeaways: 
# - All of the filter keyword are pretty similar because they were all extracted from the top 100 recipes within their topic.
# - Recipes NLP stopwords need to be compiled
# - The keywords that TextRank extracts relate really well to the underlying extracted topic, however, there's no way of abstractively generating semantically similar words that collectively captures the essence of the topic
# - False adjacencies are created but the produced keywords do not look dissimilar from the topic modelled words using NNMF.
# - One downside of using TextRank is that upon inspection of the ranked words and the filter words, it appears multiple copies of the same words are not getting identified as unique, such that, for example, there will be multiple copies of 'toothpick' or 'cups' with different individual rankings.
# 
# 

# ### Analyzing Score distribution of document and word ranks within Topics
# The purpose is to visualize the distribution of document topic rankings and decide the cutoff for the documents that associate most with that topic.

# In[429]:


import matplotlib.pyplot as plt
# text_lda
# text_nmf
# ranked


# In[456]:


# LDA Topic documents for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_lda[:,i])
    plt.subplot(1,3,i+1)
    plt.hist(series[series > 0.05])
    plt.title('LDA Topic #{} Doc Score Dist (>0.05)'.format(i+1))
plt.show()


# In[1034]:


# NNMF Topic documents for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_nmf[:,i])
    plt.subplot(1,3,i+1)
    plt.hist(series[series > 0.004])
    plt.title('NNMF Topic #{} Document Score Dist (>0.004)'.format(i+1))
    plt.xlabel('Document Topic Score')
#plt.savefig('DocsByTop_Score_Distributions.png', transparent = True)
plt.show()


# Based on the first three topics for LDA and NNMF, I will subjectly choose the top 1500-2000 documents. This number of documents seems to be where the loadings for each score distibution either level off or spike upward (an elbow so to speak).
# 
# Confirming with Topic \#1 of the NNMF topics, the baking topic, baking recipes extended as far as 10,000 of the top ranked recipes within that category.

# In[462]:


# LDA Topic document scores for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_lda[:,i]).copy().sort_values(ascending = False).reset_index(drop = True)
    plt.subplot(1,3,i+1)
    plt.plot(series[:1000])
    plt.title('LDA Topic #{} Ordered Score Plot'.format(i+1))
plt.show()


# In[1035]:


# NMF Topic document scores for topics 0-2
plt.figure(figsize=(15,4))
for i in range(3):
    series = pd.Series(text_nmf[:,i]).copy().sort_values(ascending = False).reset_index(drop = True)
    plt.subplot(1,3,i+1)
    plt.plot(series[:1000])
    plt.title('NMF Topic #{} Ordered Score Plot'.format(i+1))
    plt.xlabel('Document Rank')
    plt.ylabel('Document Topic Score')
#plt.savefig('DocsByTop_Score_Elbows.png', transparent = True)
plt.show()


# While this method has not been previously utilized, the plots above take  after scree plots, and they plot the scores of documents as they relate to their respective topics in descending order. By plotting these elbow plots, it might shed light on the best number of documents to use to rank words.
# 
# We will be using NNMF for the model because the topics seemed to have converged more distinctly on topic. According to the 'Scree Plots' corresponding to the NNMF topics, it seems that the scores begin leveling out around 200 documents. Therefore we will use TextRank on the top 200 documents for each topic.

# ***One Note: Go deeper into depth why you chose NNMF over LDA.***

# # \~~Putting it all together~~
# 
# Reintialize the model with 'pseudo-optimized' parameters, more easily track flow of data, and toggle with parameters all in one place. The "database" will also be created so that user queries will return results in a speedy manner!
# 
# Some pieces of code will be commented out with a triple ***\''' '''*** to indicate that the code takes too long to run and should only be run when the kernel has been shutdown.

# In[ ]:


import pandas as pd
import numpy as np
import re
import spacy
from functools import reduce
from operator import add
import string
import re
import multiprocessing as mp

### Below is all the code necessary to clean the data into useable form for modeling.
'''
# Loading Data
allrecipes_raw = pd.read_json('../__DATA__/recipes_raw/recipes_raw_nosource_ar.json')
epicurious_raw = pd.read_json('../__DATA__/recipes_raw/recipes_raw_nosource_epi.json')
foodnetwork_raw = pd.read_json('../__DATA__/recipes_raw/recipes_raw_nosource_fn.json')

allrecipes = allrecipes_raw.copy().T.reset_index().drop(columns = ['index'])
epicurious = epicurious_raw.copy().T.reset_index().drop(columns = ['index'])
foodnetwork = foodnetwork_raw.copy().T.reset_index().drop(columns = ['index'])
recipes = pd.concat([allrecipes, epicurious, foodnetwork]).reset_index(drop=True) # Concat does not reset indices

# Cleaning
null_recs = recipes.copy().drop(columns = 'picture_link').T.isna().any()
rows_to_drop = recipes[null_recs].index
recipes = recipes.drop(index = rows_to_drop).reset_index(drop = True)

nc_ingred_index = [index for i, index in zip(recipes['ingredients'], recipes.index) if all(j.isdigit() or j in string.punctuation for j in i)]
nc_title_index = [index for i, index in zip(recipes['title'], recipes.index) if all(j.isdigit() or j in string.punctuation for j in i)]
nc_instr_index = [index for i, index in zip(recipes['instructions'], recipes.index) if all(j.isdigit() or j in string.punctuation for j in i)]

index_list = [nc_ingred_index, nc_title_index, nc_instr_index]

inds_to_drop = set(reduce(add, index_list))
print(len(inds_to_drop))
recipes = recipes.drop(index=inds_to_drop).reset_index(drop=True)
recipes.shape

empty_instr_ind = [index for i, index in zip(recipes['instructions'], recipes.index) if len(i) < 20]
recipes = recipes.drop(index = empty_instr_ind).reset_index(drop=True)

ingredients = []
for ing_list in recipes['ingredients']:
    clean_ings = [ing.replace('ADVERTISEMENT','').strip() for ing in ing_list]
    if '' in clean_ings:
        clean_ings.remove('')
    ingredients.append(clean_ings)
recipes['ingredients'] = ingredients

recipes['ingredient_text'] = ['; '.join(ingredients) for ingredients in recipes['ingredients']]
recipes['ingredient_text'].head()

recipes['ingredient_count'] = [len(ingredients) for ingredients in recipes['ingredients']]

all_text = recipes['title'] + ' ' + recipes['ingredient_text'] + ' ' + recipes['instructions']

def clean_text(documents):
    cleaned_text = []
    for doc in documents:
        doc = doc.translate(str.maketrans('', '', string.punctuation)) # Remove Punctuation
        doc = re.sub(r'\d+', '', doc) # Remove Digits
        doc = doc.replace('\n',' ') # Remove New Lines
        doc = doc.strip() # Remove Leading White Space
        doc = re.sub(' +', ' ', doc) # Remove multiple white spaces
        cleaned_text.append(doc)
    return cleaned_text

cleaned_text = clean_text(all_text)

# Testing Strategies and Code
nlp = spacy.load('en')
' '.join([token.lemma_ for token in nlp(cleaned_text[2]) if not token.is_stop])

def text_tokenizer_mp(doc):
    tok_doc = ' '.join([token.lemma_ for token in nlp(doc) if not token.is_stop])
    return tok_doc

# Parallelzing tokenizing process
pool = mp.Pool(mp.cpu_count())
tokenized_text = pool.map(text_tokenizer_mp, [doc for doc in cleaned_text])
'''

# Creating TF-IDF Matrices and recalling text dependencies

'''import text_tokenized.csv here to'''

# TF-IDF vectorizer instance
'''vectorizer = TfidfVectorizer(lowercase = True,
                            ngram_range = (1,1))'''

'''text_tfidf = vectorizer.fit_transform(tokenized_text)'''


# In[514]:


# Set All Recommendation Model Parameters
N_topics = 50             # Number of Topics to Extract from corpora
N_top_docs = 200          # Number of top documents within each topic to extract keywords
N_top_words = 25          # Number of keywords to extract from each topic
N_docs_categorized = 2000 # Number of top documents within each topic to tag 
N_neighbor_window = 4     # Length of word-radius that defines the neighborhood for
                          # each word in the TextRank adjacency table

# Query Similarity Weights
w_title = 0.2
w_text = 0.3
w_categories = 0.5
w_array = np.array([w_title, w_text, w_categories])

# Recipe Stopwords: for any high volume food recipe terminology that doesn't contribute
# to the searchability of a recipe. This list must be manually created.
recipe_stopwords = ['cup','cups','ingredient','ingredients','teaspoon','teaspoons','tablespoon',
                   'tablespoons','C','F']


# In[515]:


# Renaming Data Dependencies
topic_transformed_matrix = text_nmf
root_text_data = cleaned_text


# ### Generating  tags (keywords/categories) and assigning to corresponding documents

# In[623]:


from itertools import repeat

#recipes['tag_list'] = [[] for i in repeat(None, recipes.shape[0])]

def topic_docs_4kwsummary(topic_document_scores, root_text_data):
    '''Gathers and formats the top recipes in each topic'''
    text_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_top_docs].index
    text_4kwsummary = pd.Series(root_text_data)[text_index]
    return text_4kwsummary

def generate_filter_kws(text_list):
    '''Filters out specific parts of speech and stop words from the list of potential keywords'''
    parsed_texts = nlp(' '.join(text_list)) 
    kw_filts = set([str(word) for word in parsed_texts 
                if (word.pos_== ('NOUN' or 'ADJ' or 'VERB'))
                and word.lemma_ not in recipe_stopwords])
    return list(kw_filts), parsed_texts

def generate_adjacency(kw_filts, parsed_texts):
    '''Tabulates counts of neighbors in the neighborhood window for each unique word'''
    adjacency = pd.DataFrame(columns=kw_filts, index=kw_filts, data = 0)
    for i, word in enumerate(parsed_texts):
        if any ([str(word) == item for item in kw_filts]):
            end = min(len(parsed_texts), i+N_neighbor_window+1) # Neighborhood Window Utilized Here
            nextwords = parsed_texts[i+1:end]
            inset = [str(x) in kw_filts for x in nextwords]
            neighbors = [str(nextwords[i]) for i in range(len(nextwords)) if inset[i]]
            if neighbors:
                adjacency.loc[str(word), neighbors] += 1
    return adjacency
                
def generate_wordranks(adjacency):
    '''Runs TextRank on adjacency table'''
    nx_words = nx.from_numpy_matrix(adjacency.values)
    ranks=nx.pagerank(nx_words, alpha=.85, tol=.00000001)
    return ranks

def generate_tag_list(ranks):
    '''Uses TextRank ranks to return actual key words for each topic in rank order'''
    rank_values = [i for i in ranks.values()]
    ranked = pd.DataFrame(zip(rank_values, list(kw_filts))).sort_values(by=0,axis=0,ascending=False)
    kw_list = ranked.iloc[:N_top_words,1].to_list()
    return kw_list

# Master Function utilizing all above functions
def generate_tags(topic_document_scores, root_text_data):
    text_4kwsummary = topic_docs_4kwsummary(topic_document_scores, root_text_data)
    kw_filts, parsed_texts = generate_filter_kws(text_4kwsummary)
    adjacency = generate_adjacency(kw_filts, parsed_texts)
    ranks = generate_wordranks(adjacency)
    kw_list = generate_tag_list(ranks)
    return kw_list

def generate_kw_index(topic_document_scores):
    kw_index = pd.Series(topic_document_scores).sort_values(ascending = False)[:N_docs_categorized].index
    return kw_index

    


# In[624]:


# Generating Tags and distributing to relevant documents
for i in range(topic_transformed_matrix.shape[1]):
    scores = topic_transformed_matrix[:,i]
    topic_kws = generate_tags(scores, root_text_data)
    kw_index_4df = generate_kw_index(scores)
    recipes.loc[kw_index_4df, 'tag_list'] += topic_kws
    if i%10 == 0:
        print('Topic #{} Checkpoint'.format(i))
print('done!')


# In[626]:


# Saving the precious dataframe so that I never have to calculate that again.
recipes.to_csv('tagged_recipes_df.csv')


# In[617]:


scores = topic_transformed_matrix[:,1]
topic_kws = generate_tags(scores, root_text_data)
kw_index_4df = generate_kw_index(scores)
recipes.loc[kw_index_4df, 'tag_list'] += topic_kws


# In[726]:


recipes.loc[:5,'tag_list']


# In[727]:


# Concatenating lists of tags into a string a collective of tags for each documents
recipes['tags'] = [' '.join(tags) for tags in recipes['tag_list']]


# In[729]:


recipes.loc[:5,'tags']


# ### Querying Algorithm
# The final product presented is a search algorithm that takes in a list of ingredients or categories, and uses the query to return relavant recipes that utilize those ingredients or are similarly related to other ingredients and those recipes.

# In[730]:


recipes.columns


# In[1124]:


# Creating TF-IDF Matrices and recalling text dependencies

'''import text_tokenized.csv here'''

# TF-IDF vectorizer instance
'''vectorizer = TfidfVectorizer(lowercase = True,
                            ngram_range = (1,1))'''

'''text_tfidf = vectorizer.fit_transform(tokenized_text)'''
# title_tfidf = vectorizer.transform(recipes['title'])
# text_tfidf    <== Variable with recipe ingredients and instructions
# tags_tfidf = vectorizer.transform(recipes['tags'])
# recipes   <== DataFrame; For indexing and printing recipes

# Query Similarity Weights
w_title = .2
w_text = .3
w_categories = .5


# In[908]:


def qweight_array(query_length, qw_array = [1]):
    '''Returns descending weights for ranked query ingredients'''
    if query_length > 1:
        to_split = qw_array.pop()
        split = to_split/2
        qw_array.extend([split, split])
        return qweight_array(query_length - 1, qw_array)
    else:
        return np.array(qw_array)

def ranked_query(query):
    '''Called if query ingredients are ranked in order of importance.
    Weights and adds each ranked query ingredient vector.'''
    query = [[q] for q in query]      # place words in seperate documents
    q_vecs = [vectorizer.transform(q) for q in query] 
    qw_array = qweight_array(len(query),[1])
    q_weighted_vecs = q_vecs * qw_array
    q_final_vector = reduce(np.add,q_weighted_vecs)
    return q_final_vector

def overall_scores(query_vector):
    '''Calculates Query Similarity Scores against recipe title, instructions, and keywords.
    Then returns weighted averages of similarities for each recipe.'''
    final_scores = title_tfidf*query_vector.T*w_title
    final_scores += text_tfidf*query_vector.T*w_text
    final_scores += tags_tfidf*query_vector.T*w_categories
    return final_scores

def print_recipes(index, query, recipe_range):
    '''Prints recipes according to query similary ranks'''
    print('Search Query: {}\n'.format(query))
    for i, index in enumerate(index, recipe_range[0]):
        print('Recipe Rank: {}\t'.format(i+1),recipes.loc[index, 'title'],'\n')
        print('Ingredients:\n{}\n '.format(recipes.loc[index, 'ingredient_text']))
        print('Instructions:\n{}\n'.format(recipes.loc[index, 'instructions']))
        
def Search_Recipes(query, query_ranked=False, recipe_range=(0,3)):
    '''Master Recipe Search Function'''
    if query_ranked == True:
        q_vector = ranked_query(query)
    else:
        q_vector = vectorizer.transform([' '.join(query)])
    recipe_scores = overall_scores(q_vector)
    sorted_index = pd.Series(recipe_scores.toarray().T[0]).sort_values(ascending = False)[recipe_range[0]:recipe_range[1]].index
    return print_recipes(sorted_index, query, recipe_range)
    


# ### Testing the Algorithm

# In[1039]:


query = ['cinnamon', 'cream', 'banana']
Search_Recipes(query, query_ranked=True, recipe_range=(0,3))


# In[1130]:


# Test Rank
query = ['wine', 'cilantro','butter']
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# ### -- Conclusions and Model Outlook --
# 
# Overall the Search_Recipes function works quite well. From experimenting with the weighting, it's clear to me that the original text of the recipes returns better results than the categories generated with TextRank. More topics need to be added; from looking at the food topic documents, it's clear that the level of granularity with which LDA and NNMF can cluster recipe is very good. Another fix for this issue is to utilize dense word embeddings that capture semantic similarities between words with more sophistication. THe biggest issue with the current model is that the words that maps to each topics or category are limited and discreet. Even if a a words is technically more related to a topic than the words extracted from the same topic, yet the word was not extracted from the topic, then the original word query won't be factored into the search through the categories.
# 
# Also it does appear that some words are more heavily weighted than others, which biases the search results towards that ingredient, although this does require more rigorous texting. "Miso" is a word that is heavily weighted in the tfidf matrices for example. One work around is to use simple rank this ingredient lower in the Search_Recipes function, but a global solution is preferable. It is perhaps more beneficial to utilize these weights that tf-idf creates, rather than finding a way to get rid of them. But experimenting with different word embeddings would be interesting.
# 
# Also, another issue is that many recipes were not assigned categories due to the model parameters, and this decreases there ranks with the text with an unfair disadvantage. Hopefully a future iteration of this model will allow all recipes to have associated categories.
# 
# Future Implementation and Changes for this model:
# 
# - Word2Vec or GloVe embeddings
# - LDA2Vec topic extraction
# - Negative Querying that decreases rank of matching queries
# - Using real databases to store data and creating a creating an user interface on which this model where this model can be easily utilized
# 
# 

# In[1044]:


# Test 
query = ['jelly','wine']
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# In[1045]:


query = ['pepper','apple','pork']
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# In[1017]:


recipes['tags'][122894]


# --------
# ### Some notes:
# List of Parameters and Evaluation Methods
# - Number of Topics
# - Number of Documents to pull keywords from
# - Number of Keywords per topic
# - Number of Documents to assign keywords to
# - Neighbor Window Size
# - Query Title Weight
# - Query Description Weight
# - Query Category Weight

# In[1047]:


### No Category Weight
query = ['cream','banana','cinnamon']
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# In[1048]:


### Empty Query
query = []
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# In[1050]:


### Only Category Weight
query = ['apple','blueberry']
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# In[1051]:


### Only Category Weight
query = ['japanese']
Search_Recipes(query, query_ranked=False, recipe_range=(0,3))


# Further Analysis:
# - Generate Tag Count column in the Recipes data frame. Analyze distribution of tags.
# - See if all of the topics are easily interpretable from the generated tags.

# ### Peerings into the generated topics

# In[1052]:


recipes.tags


# In[1053]:


recipes.tags[13]


# In[1054]:


recipes.tags[122907]


# In[1055]:


recipes.tags[90708]


# In[1057]:


recipes.tags[50409]


# In[1058]:


recipes.tags[30234]


# In[1061]:


recipes.tags[23596]


# In[1102]:


recipes.tags[60457]


# In[1117]:


recipes.tags[110997]


# In[1123]:


recipes.head()


# In[ ]:




