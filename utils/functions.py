import re
import nltk
from langdetect import detect, LangDetectException

def verify_language(dataset, column):

    non_english = []
    for row in dataset[column]:
        try:
            if detect(row) != 'en':
                non_english.append(row)
        except Exception as e:
            print(f"Error detecting language for text: {row}. Error: {e}")

    if len(non_english) == 0:
        print('All texts are in English.')
    else:
        print(f'Found {len(non_english)} non-English texts, which represent {len(non_english)/len(dataset)*100:.2f}% of the dataset.')
        print('Dropping non-English texts from the dataset...')
        dataset.drop(dataset[dataset[column].isin(non_english)].index, inplace=True)



def clean_text(dataset,column, emojis = True, hashtags = True, mentions = True,
                  urls = True, html_tags = True, square_brackets = True, newlines = True,
                  whitespaces = True, punctuation = True):

    cleaned_col = dataset[column].astype(str).copy() 
    
    if emojis:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub('[\u2600-\u27FF]|[\U00010000-\U0010ffff]',' ', x))
    
    if hashtags:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'#\S+',' ', x))
            
    if mentions:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'@\S+', ' ', x))
            
    if urls:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'https?://\S+|www\.\S+',' ', x))
            
    if html_tags:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'<.*?>',' ', x))

    if square_brackets:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'\[.*?\]', ' ', x))
            
    if newlines:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'[\n\t\r]+', ' ', x))
            
    if punctuation:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))
            
    if whitespaces:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'\s+', ' ', x).strip())

    return cleaned_col
    






def lemmatize_all(token, list_pos=["n","v","a","r","s"]):
    """Lemmatize one token with WordNet, trying each part of speech in list_pos in turn:
    n (noun), v (verb), a (adjective), r (adverb), s (satellite adjective)."""

    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token

def tokenization_processing(
                        cleaned_text,
                        print_output = True,
                        pos_tagging = False,
                        no_stopwords = True,
                        lemmatized = True,
                        list_POS = ["n","v","a","r","s"],
                        stemmed = False,
                        tokenized_output = True):

    tokenized_text = nltk.tokenize.word_tokenize(cleaned_text)
    pos_tags = [pos[1] for pos in nltk.tag.pos_tag(tokenized_text)]

    if pos_tagging:
        pos_tags = nltk.pos_tag(tokenized_text)
    else:
        pos_tags = None

    if no_stopwords:
        stopwords = set(nltk.corpus.stopwords.words("english"))
        
        custom_stopwords = {
            "im", "dont", "like", "got", "get",
            "na", "oh", "yeah", "aint", "know", 
            "go", "come", "see", "youre", "make","cant"
        }
        stopwords.update(custom_stopwords)
        

        tokenized_text = [token for token in tokenized_text if token not in stopwords]

    if lemmatized:
        tokenized_text = [lemmatize_all(token, list_pos=list_POS) for token in tokenized_text]

    if stemmed:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]
        
    if not tokenized_output:
        # Se tokenized_output=False, devolve apenas uma string limpa pronta para o Scikit-Learn
        return " ".join(tokenized_text)


    if print_output:
        print(cleaned_text)
        print(tokenized_text)

    
    return " ".join(tokenized_text)    

  