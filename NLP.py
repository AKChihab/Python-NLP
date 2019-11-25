
#Make a new column to detect the length of a text messages
#Suppose that your dataframe is called df and the feutures "message" contains the text.

#This request allows you to create a new column (the length of text)
df['length'] = df['message'].apply(len)
df.length.describe()

#To visualize and to know more about the distribution of df.length
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
df['length'].plot(bins=50, kind='hist') 


# To Delete/remove punctuation from text

import string
# Example
import string

mess = 'Hello. Sample message! Does this text has punctuation? Answer: no \ yes'

# Check characters to see if they are in punctuation
nopunct = [char for char in mess if char not in string.punctuation]
print (nopunc)
#['H', 'e', 'l', 'l', 'o', ' ', 'S', 'a', 'm', 'p', 'l', 'e', ' ', 'm', 'e', 's', 's',
#'a', 'g', 'e', ' ', 'D', 'o', 'e', 's', ' ', 't', 'h', 'i', 's', ' ', 't', 'e', 'x', 
#'t', ' ', 'h', 'a', 's', ' ', 'p', 'u', 'n', 'c', 't', 'u', 'a', 't', 'i', 'o', 'n',
#' ', 'A', 'n', 's', 'w', 'e', 'r', ' ', 'n', 'o', ' ', ' ', 'y', 'e', 's']
# Join the characters again to form the string.
nopunct = ''.join(nopunc)
print (nopunc)
#Hello Sample message Does this text has punctuation Answer no  yes



# Remove Stopwords.
#We can import a list of english stopwords from NLTK
from nltk.corpus import stopwords
# Show the ten first stop words
stopwords.words('english')[0:10]
#['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']

# Application
# Now just remove any stopwords
clean_message = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
clean_message
#['Hello', 'Sample', 'message', 'text', 'punctuation', 'Answer', 'yes']

#LET's Create a function to apply all the process above
def text_process(message):

    # 1. Remove all punctuation
    # Check characters to see if they are in punctuation
    nopunc = [char for char in message if char not in string.punctuation]

    # 2. Remove all stopwords
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # 3. Returns a list of the cleaned text
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
#Application
df['clean_mess"]=df['message'].apply(text_process)


#We can imagine a 2-Dimensional matrix where the 1-dimension is the entire vocabulary(1 row per word) 
#and the other dimension are the messages, in this case a column per text message.

For example:

            Message1	Message2	...	MessageN
Word1Count	    0	        1	    ...	  0
Word2Count	    0	        0	    ...  	0
    ...	        1	        2	    ...	  0
WordNCount     	0	        1	    ...	  1

#Since there are so many messages, we can expect a lot of zero counts for the 
#presence of that word in a text.

#APPLICATION
from sklearn.feature_extraction.text import CountVectorizer
# A lot of arguments and parameters that can be passed to the CountVectorizer.
# We can specify the analyzer to be our own previously defined function "text_process"

mess_vect = CountVectorizer(analyzer=text_process).fit(df['message'])

# Print total number of vocab words
print(len(mess_vect.vocabulary_))

#GET the word in the row K( K is a numbre)
print(mess_vect.get_feature_names()[K])

#Transform our dataframe of messages to a Matrix
mess_word = mess_vect.transform(df['message'])

#Shape of th Matrix
print('Shape of Sparse Matrix: ', mess_word.shape)
#Count of positive occurences
print('Amount of Non-Zero occurences: ', mess_word.nnz)

#Sparsity and TF-IDF
