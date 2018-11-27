
# coding: utf-8

# ### Installations, if needed

# In[ ]:


get_ipython().system("pip install spacy==2.0.12 # Above 2.0.12 doesn't seem work with the neuralcoref resolution (at least 2.0.13 and 2.0.16 don't)")


# ### Importing Libraries

# In[2]:


import spacy
from spacy import displacy
from collections import Counter
import re
import os
import pandas as pd
import sys


# ### Loading and previewing our export from OntoNotes5

# In[3]:


JSON_FILENAME = 'ner_output_1.json'
FILEPATH_TO_JSON = "onto_sql_output/"

onto_import = pd.read_json(FILEPATH_TO_JSON + JSON_FILENAME)


# In[4]:


onto_import.head()


# ### Downloading and loading the large spaCy English pipeline

# In[5]:


# Download the english medium-sized pipeline
get_ipython().system(' python -m spacy download en_core_web_lg')


# In[7]:


nlp = spacy.load('en_core_web_lg')

# Testing one sentence
doc = nlp(onto_import.loc[1,'sentence_string'])


# ### Looping through dependency parsing for all sentences

# In[12]:


onto_import["spacy_parse"] = onto_import.apply(lambda x: nlp(x["sentence_string"]), axis=1)


# In[15]:


onto_import.head()


# ### Viewing the text, with highlighted named entities

# In[35]:


# Choose an entry integer to see its text and the parse below.
ENTRY = 47

displacy.render(onto_import.loc[ENTRY,"spacy_parse"], jupyter=True, style='ent')


# ### Viewing dependencies

# In[36]:


displacy.render(onto_import.loc[ENTRY,"spacy_parse"], jupyter=True, style='dep')

