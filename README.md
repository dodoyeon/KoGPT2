# Relay Novel Generation Program : 한국어 릴레이 소설 생성 프로그램
![가계도](https://user-images.githubusercontent.com/18351404/126857974-230380fa-9531-4109-8c05-8f3764199ec0.png)

The Relay novel generation program makes ML model to receive the sentence of the novel written by the user as an input and predicts the next part of the novel. This process write a relay novel between user and program.
Using the homepage, the output is printed and many users can collaborate to create their creations.

계획한 프로젝트의 아웃풋은 유저가 쓴 소설의 일부분을 AI가 인풋으로 받아 소설의 다음 내용을 예측해 출력하여 릴레이 소설을 쓸 수 있는 프로그램이다.
홈페이지를 이용해 많은 유저들이 창작물을 협력하여 만들 수 있다.


## Creative & Self-motivating Project Class
### base Model GPT-2

One-way language model that is similar to decoder-only Transformer

It is called Auto-Regressive model that the predicted token enters the input again as shown in the figure below.
Decoder uses the masked self-attention to mask tokens except already predicted tokens.

트랜스포머 디코더모듈만 사용되는 단방향 language model

밑의 그림처럼 예측한 토큰이 다시 입력으로 들어가기 때문에 auto-regression이라고 한다.
decoder는 masked self-attention을 사용한다: 미리 앞에 나올 토큰을 보지 않도록 현재 예측한 토큰까지만 인풋으로 들어갈 수 있도록 마스킹을 해준다.

* [SKT KoGPT2 pre-trained model v2](https://github.com/SKT-AI/KoGPT2)

SKT brain에서 한국어 위키 백과 이외, 뉴스, 모두의 말뭉치 v1.0, 청와대 국민청원등 
한국어 데이터셋을 이용해 pre-training 한 GPT-2 모델을 이용하였다.


[reference]
* [Task using KoGPT-2](https://github.com/MrBananaHuman/KorGPT2Tutorial)

* [Korean novel generation GPT-2](https://github.com/shbictai/narrativeKoGPT2)


### Dataset
fine-tuning dataset: searchable, copyright expired fantasy genre novels
Web crawling : 위키문헌, etc

![데이터차이](https://user-images.githubusercontent.com/18351404/126858161-e6523ffc-fcc8-41cb-b65f-32f89a050c2a.png)
[위키 문헌](https://ko.wikisource.org/wiki/%EC%9C%84%ED%82%A4%EB%AC%B8%ED%97%8C:%EB%8C%80%EB%AC%B8)


[모두의 말뭉치](https://corpus.korean.go.kr/)


### Network
Python Flask (Web Framework)

![Homepage](https://user-images.githubusercontent.com/18351404/126859963-8044bd34-4eb9-4b04-a144-5283f50ca0c7.png)

[reference]
* [flask-restx api tutorial](https://justkode.kr/python/flask-restapi-1)

* [Flask tutorial](https://flask.palletsprojects.com/en/1.1.x/quickstart/)


### Training 
GPU : v100 100Gb
Fine tuning Dataset: About 60,000 sentences, 6MB txt file

![tr그래프](https://user-images.githubusercontent.com/18351404/126858352-de80d366-a634-41c7-a28e-0a627167125f.png)


### Result
* 1 sentence 1 input (5 epoch)

![그림1](https://user-images.githubusercontent.com/18351404/126892716-cd5b6576-dad1-4e3f-b571-b01914565b17.png)

* 1022 length paragraph 1 input (20 epoch)

![그림2](https://user-images.githubusercontent.com/18351404/126892722-9714634f-9218-486d-bbba-62a03970b7f4.png)

* 1022 length paragraph 1 input (60 epoch)

![그림3](https://user-images.githubusercontent.com/18351404/126892725-a117bca3-48c0-4774-b089-8673919ba098.png)

The generation result is not good.

generation 결과는 부진하였다.

### 부족한 점 및 문제점 인식
Limitation of SKT KoGPT-2
* limitation of Pre-trained KoGPT-2
![그림1](https://user-images.githubusercontent.com/18351404/126859763-be3e7ea2-56b8-40b3-86dd-e3d35e565568.png)

The following is sentence generation using the API provided by SKT.

다음은 SKT에서 제공하는 API 를 이용해 문장을 generation 한 것 이다.


  ![그림4](https://user-images.githubusercontent.com/18351404/126892744-b206c38c-a7f3-4701-8524-90d070a7c1ca.png)
  
The following is sentence generation using the pre-trained KpGPT-2.

다음은 pre-trained 된 model만을 이용해 문장을 generation 한 것이다.

* Insufficient fine-tuning Dataset

* Poor in homepage implementation: relay implementation failed

  홈페이지 구현 측면에서 미흡: 릴레이 구현을 못함

## OpenAI GPT-3 Api 
*  GPT-3 model fine-tuning

*  Limiting fine tuning dataset to fantasy genre caused the fine-tuning dataset
   (approximately 20M expected)
   fine tuning dataset을 판타지 장르로 한정시킨것이 문제점 : 다양한 장르를 이용해 데이터 수를 늘릴 수 있을것이다. (약 20M 는 필요할것으로 예측)

*  Hyper parameter tuning : Epoch, learning rate
