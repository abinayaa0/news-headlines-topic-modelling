# Topic Modeling

**Libraries required for topic modeling**: Pandas, gensim and pyLDAvis


# Install Libraries

!pip install PyLDAvis



!pip install -U gensim

# Import Libraries

# import dependencies
import pandas as pd
import numpy as np

#Dependencies for Data Pre-processing
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stoplist = stopwords.words('english') 
stoplist.extend(['last','updated'])
stoplist = set(stoplist)
import re
import string

#Libraries for dictionary, doc_term_matrix, LDA implementation
import gensim
import gensim.corpora as corpora
from pprint import pprint

#Libraries for Visualization
%matplotlib inline
import matplotlib.pyplot as plt

#pyldavis library helps dynamic visualization of topics
import pyLDAvis

pyLDAvis.enable_notebook()

# Read Cleaned Data with StopWords

# Input pre-processed text 
# read data
import pandas as pd
DF = pd.read_csv('bbc_data.csv')
DF.head()



step I: read the pre processed data:

#Input pre-processed text from Objective 1 of action learning plan
# write code to read data
import pandas as pd
DF = pd.read_csv('bbc_data.csv')
DF.head()
DF['Content_nGrams']= DF['Processed_Content']
Processed_Content = DF['Content_nGrams']
DF.head()



# Read n_grams 

def readFile(bbc_data):
    """
    This function will read the text files passed & return the list
    """
    fileObj = open(bbc_data, "r") #opens the file in read mode
    words = fileObj.read().splitlines() #puts the file into a list
    fileObj.close()
    return words

def read_nGrams():
    """
    This function will read bigrams & trigrams and 
    return  list of n_Grams.
    """
    fileObj = open(bbc_data, "r") #opens the file in read mode
    words = fileObj.read().splitlines() #puts the file into a list
    fileObj.close()
    # read  bigrams 
    original_bigram = readFile("bigram.txt")
    # read trigrams
    original_trigram = readFile("trigram.txt")

    # Combined list of bigrams & trigrams
    n_grams_to_use = []
    n_grams_to_use.extend(original_bigram)
    n_grams_to_use.extend(original_trigram)
    return n_grams_to_use
n_grams_to_use = read_nGrams()

# Generating combined n_Grams

# Combine each n_Gram using '_'
def combined_n_Grams(n_grams_to_use):
    """
    This function will read n_Grams & return list of combined n_Grams using '_'
    """
    Combined_nGrams = []
    for i in range(len(n_grams_to_use)):
        Combined_nGrams.append(n_grams_to_use[i].replace(' ','_'))
    return Combined_nGrams
Combined_nGrams = combined_n_Grams(n_grams_to_use) 
Combined_nGrams

# Mapping of combined n_Grams to that of individual n_Grams

def mapping(n_grams_to_use, Combined_nGrams):
    """
    This function will map combined n_Grams with that of individual n_Grams & return the dictionary.
    """
    dic=dict()
    for i in range(len(Combined_nGrams)):
        dic[n_grams_to_use[i]] = Combined_nGrams[i]
    return dic
Mapping = mapping(n_grams_to_use, Combined_nGrams)
Mapping

## Step1: Add n-grams back into the reviews

To add n-grams into the reviews. The input data has a list of n-grams generated in collocation step. They need to be replaced back into the data.




def add_ngrams_to_input(Processed_content,Mapping):
    """
    This function will replace original occurrence of n_Grams in the text with that of Combined n_Grams.
    """
    for i in range(len(Processed_content)):
        for key, value in Mapping.items():
            Processed_content[i] = Processed_content[i].replace(key, value)
    return Processed_content
content_nGrams = add_ngrams_to_input(Processed_Content,Mapping)

DF.head()

## Step2: Remove Stopwords from the input text

There is a need to remove stopwords from the input text because such words doesn't play any role in defining topics. 

def removing_stopwords(text):
    """This function will remove stopwords which doesn't add much meaning to a sentence 
       & they can be remove safely without comprimising meaning of the sentence.
    
    arguments:
         input_text: "text" of type "String".
         
    return:
        value: Text after omitted all stopwords.
        
    Example: 
    Input : This is Kajal from delhi who came here to study.
    Output : ["'This", 'Kajal', 'delhi', 'came', 'study', '.', "'"] 
    
   """
    # repr() function actually gives the precise information about the string
    text = repr(text)
    # Text without stopwords
    No_StopWords = [word for word in word_tokenize(text) if word.lower() not in stoplist]
    # Convert list of tokens_without_stopwords to String type.
    words_string = ' '.join(No_StopWords) 
    return words_string


## Step3: Removing Punctuations

I have considered some special characters (.,?!) as valid for our future work at the time of pre-processing the data, but are they really important from topic modeling point of view. Remember in topic modeling the idea is that Documents are comprised of Topics and Topics are made of words. 

def removing_special_characters(text):
    """Removing all the special characters except the one that is passed within 
       the regex to match, as they have imp meaning in the text provided.
   
    
    arguments:
         input_text: "text" of type "String".
         
    return:
        value: Text with removed special characters that don't require.
        
    Example: 
    Input : Hello, K_a_j_a_l. Thi*s is $100.05 : the payment that you will recieve! (Is this okay?) 
    Output :  Hello K_a_j_a_l This is 100 05  the payment that you will recieve Is this okay
    
   """
    # The formatted text after removing not necessary punctuations.
    
    Formatted_Text = re.sub(r"[^a-zA-Z0-9_']+", ' ', text) 
    # In the above regex expression,I am providing necessary set of punctuations that are frequent in this particular dataset.
    return Formatted_Text


## Step4: Tokenization

Breakdown text as list of tokens to create dictionary and document term matrix for topic model.
The results will a list of list of input text. 

**Resources**: wordtokenizer from nltk

def tokenize_text(Updated_content):
    """
    This function will tokenize the word after removing stopwords & punctuations 
    and return the list of list of articles.
    """
    tokenized_text = [word for word in word_tokenize(Updated_content)]
    return tokenized_text

# Writing main function to merge all the preprocessing steps.
def text_preprocessing(text,  punctuations=True,  token = True,
                       stop_words=True, apostrophe=False, verbs=False):
    """
    This function will preprocess input text and return
    the clean text.
    """
    stoplist = stopwords.words('english') 
    stoplist = set(stoplist)
    
    if stop_words == True: #Remove stopwords
        Data = removing_stopwords(text)
    
    if punctuations == True: #remove punctuations
        Data = removing_special_characters(Data)
        
    if token == True: # Tokenize text
        Data = tokenize_text(Data)  
    if apostrophe == True: #Remove apostrophes
        Data = remove_apostrophe(Data)
    if verbs == True: #Remove Verbs
        Data = remove_verbs(Data)
           
    return Data

# Pre-processing for Content
List_Content = DF['Content_nGrams'].to_list()
Final_Article = []
Complete_Content = []
for article in List_Content:
    Processed_Content = text_preprocessing(article) #Cleaned text of Content attribute after pre-processing
    Final_Article.append(Processed_Content)
Complete_Content.extend(Final_Article)
DF['Updated_content'] = Complete_Content
#print(Complete_Content)


DF.head()

# Filtering of Tokens on basis of POS_Tags

def Pos_tagging(text):
    """
    This function will tag part of speeches corresponding to every tokens in the Corpus using NLTK.
    """
    tagged_articles=[]
    for articles in text:
        tagged = nltk.pos_tag(articles)
        #print(tagged[100:150])
        tagged_articles.append(tagged)
    #print(tagged_articles)
    return tagged_articles
tagged_articles = Pos_tagging(Complete_Content)


### List of POS tags
https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

def remove_POS(text):
    """
    This function will check for all POS tags and tell about necessary & unnecessary tags.
    """
    cardinals=[]
    Modal=[]
    Adverb=[]
    Adjective=[]
    Preposition=[]
    Verb=[]
    Verbs=[]
    Verbz=[]
    POS=[]
    check=[]
    for articles in text:
        for i, (token, POS_tag) in enumerate(articles):
            if (POS_tag=='CD'):                
                cardinals.append(token) # Can be dropped
            elif (POS_tag=='MD'):
                Modal.append(token) # Can be dropped to improve results
            elif (POS_tag=='RB'):
                Adverb.append(token) # Can't drop them
            elif (POS_tag=='JJ'):
                Adjective.append(token) # Can't drop them
            elif (POS_tag=='IN'):
                Preposition.append(token) # Can't drop them
            elif (POS_tag=='VB'):
                Verb.append(token) # Can't drop them but we can provide some of the tokens in stoplist to remove.
            elif (POS_tag=='VBG'):
                Verbs.append(token) #These verbs can be dropped (unique-780)(total-6417) (need to see again)
            elif (POS_tag=='VBZ'):
                Verbz.append(token) #Can't be dropped
            elif (POS_tag=='POS'):
                POS.append(token) # Should drop it.
            elif (POS_tag=='PRP'):
                check.append(token) # Nouns can't be dropped, coordinating junction should be dropped, take a look at DT(keep imp ones & drop rest),
                #'FW'--> Imp to keep, Adjectives can't be dropped, 'PRP' should be dropped after keeping imp tokens.
            
            
    #print(set(cardinals))
    print(set(Modal))
    #print(set(Adverb))
    #print(set(Adjective))
    #print(set(Preposition))
    #print(set(check))
remove_POS(tagged_articles)
# After analysing each POS tag that tokens in my text can have, I figure out there are many tokens which are tagged by different POS_tags.
# Nouns mostly included every tokens.
# Shoud Remove CD,MD,POS,CJ,PRP,DT
# Look at imp tokens and provide rest of unnecessary tokens in a stoplist --> VB,VBG,DT,PRP

def keeping_nouns_only(text): 
    """
    This Function will keep tokens tagged with Nouns and remove everything else from the corpus
    & return with list of list of articles with filtered tokens.
    """
    Result=[]
    for i in range(len(text)):
        Articles_Nouns=[]
        for j in range(len(text[i])):
            if (text[i][j][1] == 'NN' or text[i][j][1]=='NNP' or text[i][j][1]=='NNS' or text[i][j][1] == 'NNPS'): 
                Articles_Nouns.append(text[i][j][0])
        Result.append(Articles_Nouns)    
    return Result
Result_Nouns = keeping_nouns_only(tagged_articles)
#print(Result)
# Looked at results after keeping nouns only and drop everything else. (Results didn't improve much.)

# Total no. of tokens in the corpus
tokens = []
for article in DF['Updated_content']:
    for word in article:
        tokens.append(word)
len(tokens) 

# Total no. of tokens in the NOUNS only corpus
list_tok = []
for article in Result_Nouns:
    for word in article:
        list_tok.append(word)
len(list_tok) 

print("Only Nouns Text",len(list_tok)) 
print("All Text", len(tokens))

DF['Updated_content']

# Unique tokens in the corpus
len(set(tokens))

## Step5: Create Dictionary and Document term matrix

Use the tokenized Input of data and prepare the Dictionary and Document Term Matrix. 

**Resources**: gensim

# define the function to create dictionary and document to term matrix
def create_dic_and_docterm_matrix(Complete_Content, dict_file_path, matrix_file_path):
    """
    This function will create corpus dictionary and document to term matrix
    
    Argument:
        X: tokenized text corpus
        dict_file_path: file path to save dictionary
        matrix_file_path: file path to save matrix
    returns:
        corpus dictionary and document to term matrix
    """   
    
    # Create Dictionary
    id2word_dic = corpora.Dictionary(Complete_Content)
    # Save Dictionary
    id2word_dic.save(dict_file_path)
 
    # Create Corpus
    text = Complete_Content # Query here(Should I keep the same corpus after tokenization or update with the one got after POS_tagging)
    #  Document to term Frequency
    doc_term_matrix = [id2word_dic.doc2bow(tokens) for tokens in text]
    # Save Doc-Term matrix
    corpora.MmCorpus.serialize(matrix_file_path, doc_term_matrix)

    return id2word_dic, doc_term_matrix
    
    
dict_file_path = r"C:\Users\Kajal\Desktop\Topic Modelling\dictionary.txt"
matrix_file_path = r"C:\Users\Kajal\Desktop\Topic Modelling\doc_term_matrix.txt"
dic_LDA, doc_term_matrix  = create_dic_and_docterm_matrix(Complete_Content, dict_file_path, matrix_file_path) 

# function to load dictionary and doc to term matrix from the file
def load_dict_and_docterm_matirx(dict_path, matrix_path):
    """
    This fucntion will load and return
    dictionary and doc term matrix
    
    Arguments:
        dict_path: path to corpus dictionary
        matrix_path: path to corpus document to term matrix
                    
    returns:
    dictionary and doc-term matrix
    """

    dictionary = corpora.Dictionary.load(dict_path)
    doc_term_matrix = corpora.MmCorpus(matrix_path)    
    return dictionary, doc_term_matrix

dictionary, doc_term_matrix = load_dict_and_docterm_matirx(dict_file_path, matrix_file_path)

## Step6: Prepare Topic model and generate Coherence scores

**Tips:** The model needs good memory and cores to train faster. Therefore select Chunksize paramter wisely. 


## Prepared Topic models

for k in range(2,25): # Train LDA on different values of k
    print('Round: '+str(k))
    LDA = gensim.models.ldamulticore.LdaMulticore
    ldamodel = LDA(doc_term_matrix, num_topics=k, id2word = dictionary, passes=20, iterations=100,
                   chunksize = 10000, eval_every = 10, random_state=20)
    ldamodel.save(f"ldamodel_for_{k}topics_Run_10")
    pprint(ldamodel.print_topics())

## Generate Coherence Score

coherence = []
for k in range(2,25):
    LDA = gensim.models.ldamulticore.LdaMulticore
    ldamodel = LDA.load(f"ldamodel_for_{k}topics_Run_10")
    cm = gensim.models.coherencemodel.CoherenceModel(model=ldamodel, texts=Complete_Content, dictionary=dictionary, coherence='c_v')
    coherence.append((k, 'default', 'default', cm.get_coherence()))

pd.DataFrame(coherence, columns=['LDA_Model','alpha','eta','coherence_score']).to_csv('coherence_matrix_10.csv', index=False)

mat = pd.read_csv('coherence_matrix_10.csv')
mat.reset_index(drop=True)
mat

# Visualize Coherence score for top 25 LDA models

# Show graph
x = range(2,25)
plt.plot(x, mat['coherence_score'])
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show() # Num Topics = 4 is having highest coherence score.

LDA = gensim.models.ldamulticore.LdaMulticore
ldamodel = LDA.load(f"ldamodel_for_16topics_Run_10")
pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)


 # Finding the dominant topic in each Article

def finding_dominant_topic(ldamodel, corpus, tokenized_content, content_nGrams, Cleaned_text):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    sent_topics_df = pd.concat([sent_topics_df, tokenized_content, content_nGrams, Cleaned_text], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = finding_dominant_topic(ldamodel=ldamodel, corpus=doc_term_matrix, tokenized_content=DF['Updated_content'], content_nGrams = DF['Content_nGrams'], Cleaned_text=DF['Processed_Content'] )

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Article_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Processed_tokenized_Text','Text_nGrams' ,'Original_Cleaned_Text']

# Show
df_dominant_topic

# Each Topic distribution across  Articles

#STORING ALL THE DATAFRAMES AS VALUES IN A DICTIONARY WHOSE KEYS ARE THE CORRESPONDING TOPICS
dictionary_of_DataFrames={}

grp=df_dominant_topic.groupby('Dominant_Topic')

#A GROUP OBJECT WILL HAVE TWO COMPONENTS. ONE:THE VALUE OF THE ATTRIBUTE ON WHICH THE DATASET IS GROUPED, TWO: THE CRRESPONDING GROUPS FOR EACH UNIQUE VALUE OF THAT ATTRIBUTE.
for topics, dataframes in grp:     
    dictionary_of_DataFrames[topics]=pd.DataFrame(dataframes[['Article_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Processed_tokenized_Text','Text_nGrams' ,'Original_Cleaned_Text']] ,columns=['Article_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Processed_tokenized_Text','Text_nGrams' ,'Original_Cleaned_Text']).reset_index(drop=True)
    
dictionary_of_DataFrames[8.0] #Details of Articles corresponding to Topic 7.

# Create csv files for each Topic representing all the corresponding articles.
pd.DataFrame(dictionary_of_DataFrames[15.0], columns=['Article_No', 'Topic_Perc_Contrib', 'Keywords', 'Processed_Text', 'Original_Text']).to_csv('Topic_16.csv', index=False)

## Summary of  understanding of topic modeling

- Results with default values of alpha & eta, random state=20 (good)
- Results with alpha=0.1 and eta = 0.01 and random state = 123. (Bad)
- Results after cleaning verbs & apostrophe marks, Keeping Nouns only (Not Improved)
- Results with default values of alpha & eta, random state=20, chunksize=10000 (Better Results to keep)

- Topic modelling refers to the task of identifying topics that best describes a set of documents. And the goal of LDA is to map all the documents to the topics in a way, such that the words in each document are mostly captured by those imaginary topics.

- LDA can hardly run on big data due to memory/time issues. It will run better if you have access to machine with 64x architecture and big RAM capacity, 16 GB or more.

- Values of lambda that are very close to 0 will show terms that are more specific for a chosen topic. Meaning that we will see terms that are "important" for that specific topic but not necessarily "important" for the whole corpus.

- Values of lambda that are very close to 1 will show those terms that have the highest ratio between frequency of the terms for that specific topic and the overall frequency of the terms from the corpus.
