import pandas as pd



def get_sentiments(column, analyzer):

    column_str = str(column)

    sentiments_scores = analyzer.polarity_scores(column_str)
    compound = sentiments_scores['compound']

    sentence = column_str.split(" ") 
    compound_scores = [analyzer.polarity_scores(sentence)["compound"] for word in sentence if word.strip()]
    mean_compound_score = sum(compound_scores) / len(compound_scores) if compound_scores else 0.0
    
    return compound_scores, mean_compound_score


def analyze_w_vader(dataset, column = 'lyrics_normalized', analyzer = None):
    compound_scores, mean_scores = zip(*dataset[column].apply(lambda x: get_sentiments(x, analyzer)))

    dataset["Compound Vader"] = compound_scores
    dataset["Mean Compound Vader"] = mean_scores
    
    return dataset