import re
import nltk
from langdetect import detect, LangDetectException

def verify_language(dataset, column):

    non_english_idx = []
    for idx, row in dataset[column].items():
        try:
            if detect(row) != 'en':
                non_english_idx.append(idx)
        except Exception as e:
            print(f"Error detecting language for text: {row}. Error: {e}")

    if len(non_english_idx) == 0:
        print('All texts are in English.')
    else:
        print(f'Found {len(non_english_idx)} non-English texts, which represent {len(non_english_idx)/len(dataset)*100:.2f}% of the dataset.')
        print('Non-English texts:')
        print(dataset.loc[non_english_idx[:5], column])
        print('Dropping non-English texts from the dataset...')
        dataset.drop(index=non_english_idx, inplace=True)




def clean_text(dataset,column, emojis = True, hashtags = True, mentions = True,
                  urls = True, html_tags = True, square_brackets = True, newlines = True,
                  whitespaces = True, apostrophes= True,punctuation = True):

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
        
    if apostrophes:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r"[']", '', x))
               
    if punctuation:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))
            
    if whitespaces:
        cleaned_col = cleaned_col.apply(
            lambda x: re.sub(r'\s+', ' ', x).strip())

    return cleaned_col


def lemmatize_function(token, list_pos=["n","v","a","r","s"]):
    """Lemmatize one token with WordNet, trying each part of speech in list_pos in turn:
    n (noun), v (verb), a (adjective), r (adverb), s (satellite adjective)."""

    wordnet_lem = nltk.stem.WordNetLemmatizer()
    for arg_1 in list_pos:
        token = wordnet_lem.lemmatize(token, arg_1)
    return token


def normalize_text(text,print_output = False, normalize_contractions = True,
                   lowercase = True, remove_stopwords = True,include_pos_tags = False,
                   custom_stopwords = None, lemmatize = True, stem = False,
                   list_pos = ["n","v","a","r","s"], tokenized_output = False):

    text = str(text)

    if lowercase:
        text = text.lower()

    if normalize_contractions:
        contractions = {
            "aint": "is not", "arent": "are not", "cant": "cannot",
            "couldve": "could have", "couldnt": "could not", "didnt": "did not",
            "doesnt": "does not", "dont": "do not", "hadnt": "had not",
            "hasnt": "has not", "havent": "have not", "wont": "will not",
            "wouldve": "would have","i m":"i am", "im": "i am", "ive": "i have", "isnt": "is not",
            "shouldnt": "should not", "wasnt": "was not","were": "we were", "werent": "were not",
            "youre": "you are", "youve": "you have", "id": "i would", "ill": "i will",
            "youll": "you will", "weve":"we have", "theyre": "they are", "theyve":"they have",
            "itll": "it will", "itd": "it would", "wan na": "want to", "gon na": "going to",
            "wanna": "want to", "gonna": " going to", "gotta": "got to", "getta": "got to",
            "got ta": "got to", "get ta":"got to"}
        
        words = text.split()
        text = " ".join([contractions.get(word, word) for word in words])

    if custom_stopwords is None:
            custom_stopwords = set()
    else:
            custom_stopwords = set(custom_stopwords)

    tokenized_text = nltk.tokenize.word_tokenize(text)
    
    if include_pos_tags:
        pos_tags = nltk.pos_tag(tokenized_text)
    else:
        pos_tags = None

    all_stopwords = set()
    if remove_stopwords:
            all_stopwords = set(nltk.corpus.stopwords.words("english"))
            if custom_stopwords:
                all_stopwords.update(custom_stopwords)

    if lemmatize:
            tokenized_text = [lemmatize_function(token, list_pos=list_pos) for token in tokenized_text]

    if remove_stopwords:
        tokenized_text = [token for token in tokenized_text if token not in all_stopwords]
        
    if stem:
        porterstemmer = nltk.stem.PorterStemmer()
        tokenized_text = [porterstemmer.stem(token) for token in tokenized_text]

    if print_output:
        print('Final normalized text:', tokenized_text)

    if tokenized_output:
        return tokenized_text
    else:
        return " ".join(tokenized_text)
    


