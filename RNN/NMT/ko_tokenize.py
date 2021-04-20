# 형태소 분석기
from konlpy.tag import Kkma
kkma = Kkma()

stems = kkma.morphs('롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.')
print(stems)

# t_text = list(open("data/korean-english-park.train.ko", "r", encoding='UTF8').readlines())
# with open("./data/korean-english-park.train_stem.ko", "w", encoding='UTF8') as f:
#     for sent in t_text:
#         print(sent)
#         stems = kkma.morphs(sent)
#         print(stems)
#         f.write(" ".join(stems)+"\n")
#
# t_text = list(open("data/korean-english-park.dev.ko", "r", encoding='UTF8').readlines())
# with open("./data/korean-english-park.dev_stem.ko", "w", encoding='UTF8') as f:
#     for sent in t_text:
#         print(sent)
#         stems = kkma.morphs(sent)
#         print(stems)
#         f.write(" ".join(stems) + "\n")
