# 한국어 릴레이 소설 프로그램 만들기
![가계도](https://user-images.githubusercontent.com/18351404/126857974-230380fa-9531-4109-8c05-8f3764199ec0.png)


## 사용Model GPT-2
Transformer decoder 만 사용되는 단방향 language model 
밑의 그림처럼 예측한 토큰이 다시 입력으로 들어가기 때문에 auto-regression이라고 한다.
decoder는 masked self-attention을 사용한다: 미리 앞에 나올 토큰을 보지 않도록 현재 예측한 토큰까지만 인풋으로 들어갈 수 있도록 마스킹을 해준다.

[SKT KoGPT2 pre-trained 모델 v2](https://github.com/SKT-AI/KoGPT2)
SKT brain에서 만든 한국어 GPT-2 모델을 이용

[KoGPT-2 활용 task](https://github.com/MrBananaHuman/KorGPT2Tutorial)


[한국어 소설 생성 GPT-2](https://github.com/shbictai/narrativeKoGPT2)


## Dataset
fine-tuning dataset: 검색 가능한, 저작권기간이 만료된 판타지 장르의 소설 이용
Web crawling : 위키문헌 등

## Network
파이썬 Flask (Python으로 구동되는 웹 프레임워크)
[flask-restx api 사용법](https://justkode.kr/python/flask-restapi-1)
[Flask tutorial](https://flask.palletsprojects.com/en/1.1.x/quickstart/)



## Training 
사용 리소스: v100 100Gb
