import re
import nltk


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
        stopwords = nltk.corpus.stopwords.words("english")
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

    return (tokenized_text, pos_tags)