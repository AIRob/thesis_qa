import jieba as jb
import json


stop_words = None

def read_txt_lines(path):
    lines = []
    with open(path,'r',encoding="utf-8") as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    return lines
    
def read_json_utf8(path):
    with open(path,'r',encoding="utf-8") as f:
        obj = json.load(f)
    return obj

def write_json_utf8(path,obj):
    with open(path,'w',encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)

def remove_stop_words(list_of_words,stop_words):
    ret = []
    for word_list in list_of_words:
        ret.append( [w for w in  word_list if w not in stop_words] )
    return ret

def get_word_list(list_of_sentences,stop_words=None):
    list_of_words = [ jb.cut(sentence , cut_all=False) for sentence in list_of_sentences]
    if stop_words is not None:
        return remove_stop_words(list_of_words,stop_words)
    return list_of_words



