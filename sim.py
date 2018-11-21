from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




class TfidfWrapper():
    def __init__(self,list_of_wordlist):
        corpus = [" ".join(wordlist) for wordlist in list_of_wordlist ] 
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(corpus)

    def get_vocab(self):
        return self.vectorizer.get_feature_names()

    def get_tfidf_weights(self):
        return self.tfidf.toarray()

    def get_word_by_id(self,word_id):
        return  self.vectorizer.get_feature_names()[word_id] 

    def get_wordid(self,word):
        return self.vectorizer.vocabulary_.get(word)
