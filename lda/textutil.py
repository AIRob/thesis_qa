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

def get_word_list(s,remove_stopword=True):
    global stop_words 
    if stop_words is None:
        stop_words =  read_txt_lines('stopwords.txt')
    words = jb.cut(s, cut_all=False)
    if remove_stopword:
        words = [w for w in words if w not in stop_words and len(w)>1] 
    return words
