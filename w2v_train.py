from gensim.models.keyedvectors import KeyedVectors
# from gensim.test.

model = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin.gz", binary=True, limit=30000) #binary=True  bin형태로 압축 돼어있는 파일을 그 형태로 읽겠다는 의미

# print(model['apple'])
#
# print("similarity between banana and fruit: {}".format(model.similarity("banana", "fruit")))

# print("similarity king and queen: {}".format(model.similarity("king", "queen")))
# print("similarity women and man: {}".format(model.similarity("women", "man")))
# print("similarity boy and girl: {}".format(model.similarity("boy", "girl")))
# print("similarity man and boy: {}".format(model.similarity("man", "boy")))
# print("similarity women and girl: {}".format(model.similarity("women", "girl")))
#
# print(model.most_similar("truck", topn=5))
# print(model.most_similar("banana", topn=5))
# print(model.most_similar("sum", topn=5))
# print(model.most_similar(positive=['king', 'women'], negative=['man'], topn=5))
print(model.most_similar(positive=['car', 'ocean'], negative=['wheel'], topn=5))
