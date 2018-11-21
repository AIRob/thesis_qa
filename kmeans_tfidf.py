from dataset.faq import QaOperation,read_qa_pair
from sim import  TfidfWrapper
from cluster import kmeans, cluster_by_labels,hierachical_cluster

qa_tuples = read_qa_pair('faq.json')
qa_op = QaOperation(qa_tuples,'stopwords.txt')
question,question_words = qa_op.get_questions(),qa_op.get_question_word_list()
tfidf_wrapper =  TfidfWrapper(question_words)
vocab = tfidf_wrapper.get_vocab()
#print(vocab)
tfidf = tfidf_wrapper.get_tfidf_weights()
triples = list(zip(question,question_words,tfidf))
k = 5
ans = hierachical_cluster(tfidf,k)
sentence_cluster = cluster_by_labels(question,ans,k)
print(sentence_cluster)

with open('hierachical_5.txt','w',encoding='utf-8') as f:
    for i,cluster in enumerate(sentence_cluster):
        print('cluster %d'%(i),file=f)
        print('- - -'*10,file=f)
        for sentence in cluster:
            print(sentence,file=f)
        print('',file=f)
        
