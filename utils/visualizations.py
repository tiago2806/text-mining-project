
import os
import ast

import matplotlib.pyplot as plt
import numpy as np


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