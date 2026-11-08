
import os
import ast
import pandas as pd
from collections import Counter, defaultdict
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


def word_cloud_generator(
    folder_path,
    dataset,
    matrix,
    word_list,
    wc,
    vectorisation="bow",
    prefix="WC_",
):
    """Save one word cloud per genre using labels and a sparse feature matrix."""
    if len(dataset) != matrix.shape[0]:
        raise ValueError("The dataset and feature matrix must have the same number of rows.")
    if len(word_list) != matrix.shape[1]:
        raise ValueError("The word list must match the feature matrix columns.")

    os.makedirs(folder_path, exist_ok=True)
    genres = dataset["tag"].dropna().unique()

    for genre in genres:
        positions = np.flatnonzero(dataset["tag"].eq(genre).to_numpy())
        mean_weights = np.asarray(matrix[positions].mean(axis=0)).ravel()
        frequencies = {
            word: float(weight)
            for word, weight in zip(word_list, mean_weights)
            if weight > 0
        }
        if not frequencies:
            raise ValueError(f"No positive feature weights were found for genre {genre!r}.")

        wc.generate_from_frequencies(frequencies)
        file_name = f"{prefix}{genre.upper()}_{vectorisation.upper()}.png"
        wc.to_file(os.path.join(folder_path, file_name))


def show_saved_images(file_names, folder_path="genre_word_clouds", panel_size=5):
    """Show saved images side by side, one panel per file, titled with its file name."""
    if not file_names:
        raise ValueError("At least one image file name must be provided.")

    fig, axes = plt.subplots(
        1,
        len(file_names),
        figsize=(panel_size * len(file_names), panel_size),
        squeeze=False,
    )
    for ax, file_name in zip(axes.ravel(), file_names):
        ax.imshow(plt.imread(os.path.join(folder_path, file_name)))
        ax.set_title(file_name)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def get_clean_words(val):
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        if val.startswith(("[", "(")):
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return val.split()
        else:
            return val.split()
    if isinstance(val, tuple):
        val = val[0] 
        
    words = []
    if isinstance(val, list):
        for x in val:
            if isinstance(x, (list, tuple)):
                words.append(str(x[0]))
            else:
                words.append(str(x))
    return words

def extract_words(item):
    if isinstance(item, str) and item.strip().startswith('['):
        try:
            item = ast.literal_eval(item)
        except (ValueError, SyntaxError):
            pass
            
    if isinstance(item, str):
        return [item]
    
    if hasattr(item, '__iter__'):
        words = []
        for sub_item in item:
            words.extend(extract_words(sub_item))
        return words
        
    return [str(item)]


        
    return [str(item)]






# COOCORUENECE MATRIX VISUALIZATION
def cooccurrence_matrix_sentence_generator(preproc_sentences, top_n=30):
    """Count sentence-level co-occurrences for the top_n most frequent words."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    sentence_tokens = [list(dict.fromkeys(sentence)) for sentence in preproc_sentences]
    word_frequencies = Counter(
        word for sentence in sentence_tokens for word in sentence
    )
    unique_words = [word for word, _ in word_frequencies.most_common(top_n)]
    if not unique_words:
        return pd.DataFrame(dtype=int)

    vocabulary = set(unique_words)
    co_occurrences = defaultdict(Counter)
    for sentence in tqdm(sentence_tokens, desc="Calculating co-occurrences"):
        selected_words = [word for word in sentence if word in vocabulary]
        for token_1 in selected_words:
            for token_2 in selected_words:
                if token_1 != token_2:
                    co_occurrences[token_1][token_2] += 1

    co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)
    word_index = {word: idx for idx, word in enumerate(unique_words)}
    for word, neighbors in co_occurrences.items():
        for neighbor, count in neighbors.items():
            co_matrix[word_index[word], word_index[neighbor]] = count

    co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)
    order = co_matrix_df.sum(axis=1).sort_values(ascending=False).index
    return co_matrix_df.loc[order, order]


def plot_genre_cooccurrence(genre_name, texts, labels, top_n=20):
    """
    Gera a matriz de co-ocorrência e o Heatmap igual ao do professor, 
    mas otimizado para as top N palavras de um género musical.
    """
    print(f"A preparar dados para o género: {genre_name.upper()}...")
    
    if not texts.index.equals(labels.index):
        raise ValueError("texts and labels must have matching indexes.")

    # Count co-occurrences only among training lyrics with the selected genre label.
    genre_texts = texts.loc[labels.eq(genre_name)]
    preproc_sentences = [text.split() for text in genre_texts.dropna().astype(str)]
    co_matrix_df = cooccurrence_matrix_sentence_generator(preproc_sentences, top_n=top_n)
    if co_matrix_df.empty:
        raise ValueError(f"No processed lyrics found for genre {genre_name!r}.")
    
    # 5. GERAR O HEATMAP (Exatamente igual à imagem do stor)
    # Criar máscara para o triângulo superior
    mask = np.triu(np.ones_like(co_matrix_df, dtype=bool))
    
    plt.figure(figsize=(20, 20))
    # sns.heatmap com a paleta Greens e a mask do professor
    sns.heatmap(co_matrix_df, cmap="Greens", mask=mask, annot=False)
    plt.title(f"Co-occurrence Matrix Heatmap: {genre_name.upper()} (Top {top_n} Words)", fontsize=20, fontweight='bold')
    plt.show()
    
    return co_matrix_df



## 2d visualizations pCA and truncatedsvd

def d2_embeddings_plotter(embeddings_reduction_df, nr_tokens, dim_red="PCA"):
    word_vector_df_slice = embeddings_reduction_df[:nr_tokens]

    x_col = f"{dim_red}_dim_1"
    y_col = f"{dim_red}_dim_2"
    fig = px.scatter(
        word_vector_df_slice,
        x=x_col,
        y=y_col,
        title=f"2D {dim_red} Projection of Word Embeddings (Top {nr_tokens} Words)"
    )

    for _, row in word_vector_df_slice.iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=row["Words"],
            showarrow=False,
            yshift=10,
            font=dict(size=9)
        )

    config = {'scrollZoom': True}
    fig.show(config=config)