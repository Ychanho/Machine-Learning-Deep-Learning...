from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences

# # Word2Vec모델 학습
# sentences = PathLineSentences("./data/news1.txt")   # 불러와서 전처리
# model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
# # size 조정 300까지도 씀 , min_count=@ @번 이상 등장한 단어들만 학습에 사용
# model.save("word2vec_from_news1.model")
# print(len(model.wv.vocab))

# 학습된 Word2Vec모델 불러와서 확인
model = Word2Vec.load("word2vec_from_news1.model")
# print(model.wv.most_similar("car", topn=5))
# print(model.wv.most_similar("apple", topn=5))
# print(model.wv.most_similar("son", topn=5))
# print("similarity between apple and fruit: {}".format(model.similarity("apple", "fruit")))
# print("similarity between apple and car: {}".format(model.similarity("apple", "car")))
print("similarity king and queen: {}".format(model.similarity("king", "queen")))
print("similarity women and man: {}".format(model.similarity("women", "man")))
print("similarity boy and girl: {}".format(model.similarity("boy", "girl")))
print("similarity man and boy: {}".format(model.similarity("man", "boy")))
print("similarity women and girl: {}".format(model.similarity("women", "girl")))
print(len(model.wv.vocab))
print(model.wv.most_similar(positive=['king', 'women'], negative=['man'], topn=10))

score, predictions = model.wv.evaluate_word_analogies('./data/analogy_task.txt')
print(score)

