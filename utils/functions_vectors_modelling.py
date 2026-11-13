import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd

def evaluate_models(X_train_vec, X_test_vec, y_train, y_test, representation_name):
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'SGD Classifier (SVM)': SGDClassifier(class_weight='balanced', loss='hinge', max_iter=1000),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=50, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # O Naive Bayes rebenta se houver valores negativos (Word2Vec / GloVe)
        if name == 'Naive Bayes' and (X_train_vec < 0).any():
            print(f"A saltar {name} (não aceita valores negativos nas representações densas).")
            continue
            
        print(f"A treinar {name} com {representation_name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        score = f1_score(y_test, preds, average='macro')
        
        results.append({
            'Representação': representation_name,
            'Modelo': name,
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