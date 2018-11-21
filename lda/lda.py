import textutil
import numpy as np
import pandas as pd
from gensim.models import ldamodel
from gensim.corpora import Dictionary

class QACorpus():
    def __init__(self,path):
        self.path = path
        self.qa_pair_list = []
        self._load()
    def _load(self):
        qa_list = textutil.read_json_utf8(self.path)
        for qa_dict in qa_list:
            question,answer = qa_dict["question"], qa_dict["ans"]
            self.qa_pair_list.append(QAPair(question,answer))
    def print_qas(self):
        for qa in self.qa_pair_list:
            print("question: %s\n %s \nanswer: %s\n"%(qa.question,qa.question_words,qa.answer))
            print('- - -'*20)
        print("Total %d qa in corpus"%(len(self.qa_pair_list)))

class QAPair():
    def __init__(self,question,answer):
        self.question = question
        self.answer = answer
        self.question_words = textutil.get_word_list(self.question)

class TopicModel():
    def __init__(self):
        pass
    # text : list of words::list    
    def lda(self,texts,topic_num=3):
        dictionary = Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        np.random.seed(1) # setting random seed to get the same results each time.
        lda_model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=topic_num,iterations=200) 
        top_words_per_topic = []
        for t in range(lda_model.num_topics):
            top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn =10)])
        df = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P'])
        print(df)

corpus = QACorpus("food_faq.json")
#corpus.print_qas()
questions  = [ qa.question_words for qa in corpus.qa_pair_list]
topic_model = TopicModel()
topic_model.lda(questions)