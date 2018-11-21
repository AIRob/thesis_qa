from textutil import  read_json_utf8,get_word_list,get_word_list,read_txt_lines



def read_qa_pair(path):
    obj = read_json_utf8(path)
    qa_tuples = list(map(lambda x:(x["question"],x["ans"]),obj))
    return qa_tuples

#class TextField():
#    def __init__(self,list_of_text):
#        self.list_of_text = list_of_text


class QaOperation():
    def __init__(self,qa_tuples,stopword_path):
        self.qa_tuples = qa_tuples
        if stopword_path is not None:
            self.stopwords = read_txt_lines(stopword_path)
    def get_questions(self):
        return list(map(lambda x:x[0],self.qa_tuples))
    def get_answers(self):
        return list(map(lambda x:x[1],self.qa_tuples))
    def get_question_word_list(self):
        questions = self.get_questions()
        return get_word_list(questions,self.stopwords)
    def get_answer_word_list(self):
        answers = self.get_answers()
        return get_word_list(answers,self.stopwords)   