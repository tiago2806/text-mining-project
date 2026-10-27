

import ast

def get_clean_words(val):
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except:
            return []
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