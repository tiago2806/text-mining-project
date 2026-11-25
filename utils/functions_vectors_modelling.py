import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
from scipy import sparse
from collections import Counter

from tqdm.auto import tqdm


def evaluate_models(X_train_vec, X_test_vec, y_train, y_test, representation_name):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state= 42),
        'SGD Classifier (SVM)': SGDClassifier(class_weight='balanced', loss='log_loss', max_iter=1000),
        #'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # O Naive Bayes rebenta se houver valores negativos (Word2Vec / GloVe)
        if name == 'Naive Bayes':

            if sparse.issparse(X_train_vec):
                has_negative = (X_train_vec.data < 0).any()
            else:
                has_negative = (X_train_vec < 0).any()

            if has_negative:
                print(f"A saltar {name}: existem valores negativos.")
                continue
            
            
        print(f"Training {name} with {representation_name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        score = f1_score(y_test, preds, average='macro')
        
        results.append({
            'Representation': representation_name,
            'Model': name,
            'Macro F1-Score': round(score, 4)
        })
        
    return pd.DataFrame(results).sort_values(by='Macro F1-Score', ascending=False)

def document_vector(word2vec_model, doc):
    """
    Recebe uma música (lista de palavras) e o modelo de embeddings.
    Devolve a média dos vetores das palavras. Se a palavra não existir, ignora.
    """
    # Filtra as palavras da música que existem no vocabulário do modelo
    words_in_vocab = [word for word in doc if word in word2vec_model.key_to_index]
    
    if not words_in_vocab:
        # Se a música estiver vazia ou com palavras estranhas, devolve zeros
        return np.zeros(word2vec_model.vector_size)
    
    # Faz a média dos vetores dessas palavras
    return np.mean(word2vec_model[words_in_vocab], axis=0)

def create_cooccurrence_matrix(texts, max_vocab_size: int = 520, window_size: int = 8):
    
    lyrics = pd.Series(texts).fillna("").astype(str)

    word_counts = Counter(
        word
        for lyric in lyrics
        for word in lyric.split()
    )
    cooccurrence_vocab = [word for word, _ in word_counts.most_common(max_vocab_size)]
    word_to_id = {word: index for index, word in enumerate(cooccurrence_vocab)}

    vocab_len = len(cooccurrence_vocab)
    cooccurrence_matrix = np.zeros((vocab_len, vocab_len), dtype=np.float64)

    for lyric in tqdm(lyrics, desc="Counting global word co-occurrences"):
        tokens = lyric.split()
        for position, word in enumerate(tokens):
            word_id = word_to_id.get(word)
            if word_id is None:
                continue

            end = min(position + window_size + 1, len(tokens))
            for context_word in tokens[position + 1:end]:
                context_id = word_to_id.get(context_word)
                if context_id is not None and context_id != word_id:
                    cooccurrence_matrix[word_id, context_id] += 1
                    cooccurrence_matrix[context_id, word_id] += 1

    cooccurrence_df = pd.DataFrame(
        cooccurrence_matrix,
        index=cooccurrence_vocab,
        columns=cooccurrence_vocab,
    )

  
    print(f"Matrix shape: {cooccurrence_matrix.shape}")
    print(f"Vocabulary size: {vocab_len}")

    return cooccurrence_matrix, cooccurrence_df, cooccurrence_vocab, word_to_id



def average_glove_vectors(texts, glove_lookup):
    glove_vector_size = len(next(iter(glove_lookup.values())))
    document_vectors = []

    for text in pd.Series(texts).fillna("").astype(str):
        vectors = [glove_lookup[token] for token in text.split() if token in glove_lookup]
        if vectors:
            document_vectors.append(np.mean(vectors, axis=0))
        else:
            document_vectors.append(np.zeros(glove_vector_size))

    return np.vstack(document_vectors)