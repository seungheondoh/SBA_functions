# SBA_functions
본 레파지토리는 유저들의 input pharagrah(str)을 받아서, 해당 pharagrah을 `키워드 추출`, `형태소 분석`, `유사도 측정` 모듈을 거쳐서, 해당 input pharagrah(str)과 유사한 Tag와 Song을 Retrieval 합니다.

### Detail Method
* keyword_extractor
    - pharagrah 내 키워드 추출
* tokenizer
    - 추출된 키워드의 형태소 (명사) 어휘만 재 추출
* multiquery_retrieval
    - 추출된 키워드를 통한 Similarity Search Model

### Folder
해당 레파지토리의 디렉토리는 다음과 같습니다.
- function.py : Detail Method 구현 문서
- requirements.txt : 사용 파이썬 라이브러리 정보
- static/user_dict.txt : 형태소 분석 시 고유어 인식 사전
- static/models : 워드 임베딩 모델/백터
- static/audio_meta/meta.json : 오디오 메타데이터
- static/audio_meta/audio : 오디오 wav,mp3 데이터

```sh
├── function.py
├── requirements.txt
└── static
    ├── user_dict.txt
    ├── models
    └── audio_meta
        ├── meta.json
        └── audio
            └── 1....
```

#### Requirements

- python3.7
- gensim==3.8.3
- JPype1==0.7.5
- konlpy==0.5.2
- numpy==1.19.0
- pandas==1.0.3
- six==1.15.0
- yake==0.3.7

``` bash
pip install -r requirements.txt
```


### Detail Method Example
* keyword_extractor
    - pharagrah 내 키워드 추출
    - input : Pharagraph (str)
    - output : Keyword List (List of str)
* tokenizer
    - 추출된 키워드의 형태소 (명사) 어휘만 재 추출
    - input : Keyword Sentence (str)
    - output : Token List (List of str)
* multiquery_retrieval
    - input 
        - Gensim Word Vector Class (Gensim Class)
        - model_input : Token List (List of str)
        - restrict_vocab : Search Tokens, Tag or Song (List of str)
    - output : Most Similar restrict_vocab, Tag or Song (List of str)

```python
import os
import json
import functions as F
from gensim.models.keyedvectors import KeyedVectors

sentence = """
    김조원 청와대 민정수석의 아파트 매도 호가 논란에 대해 미래통합당 하태경 의원이 “청와대에 아내 핑계 매뉴얼이라도 있느냐”고 비꼬았다. 하 의원은 7일 자신의 페이스북에 “청와대 고위 관계자가 김조원 민정수석의 고가 아파트 매물 논란에 대해 남자들은 부동산 거래 잘 모른다는 해명을 내놨다”며 “문재인 정부 남자들은 불리하면 하나같이 아내 핑계를 댄다”고 적었다. 그는 이어 조국 전 민정수석과 김의겸 전 대변인 사례를 들며 “청와대에 불리하면 아내 핑계 대라는 대응 매뉴얼이라도 있는 것이냐”며 “‘남자들은 부동산 모른다’는 청와대 관계자의 발언은 투기꾼들은 모두 여자라는 주장인지 되묻고 싶다”고 밝혔다. 하 의원은 또 “청와대에 남으려면 2주택을 무조건 팔아야 하는 소동도 괴상하지만 일단 국민에게 약속했다면 당사자인 김 수석이 책임지고 지켜야 한다”며 “자기 부동산 하나 맘대로 못해 아내 핑계 대는 사람은 국정 맡을 자격도 없다”고 주장했다. 앞서 서울 강남 지역에 아파트 2채를 보유한 김 수석은 한 채를 주변 시세보다 1~2억원 비싼 가격에 매물로 내놨다가 다주택을 처분할 뜻이 없는 것 아니냔 비판을 받았다. 이후 청와대 고위 관계자는 “통상 부동산 거래를 할 때 남자들은 잘 모르는 경우가 있다”고 해명해 논란을 키웠다.
    """
keyword_ex= F.keyword_extractor(sentence)
print(keyword_ex)
pos_tokens = F.tokenizer(" ".join(keyword_ex))
model_input = list(set(pos_tokens))
print(model_input)

model = KeyedVectors.load('./static/models/model', mmap='r')
audio_meta = json.load(open('./static/audio_meta/meta.json', 'r'))
tag_set = ['Ambient','R & B','SF','Vlog','World','가족','감동','고양','광고','광고하는','교육','그루비',
            '극도의','기술','냉기','뉴스','뉴에이지','다큐멘터리','댄스','동양적인','드라마','드럼 &베이스','듣기 쉬운',
            '라이프 스타일','라틴','락','레저','로맨틱','메탈','무서운','범죄','블로그','블루스','비디오 게임','섹시한',
            '소름','소울','쉬운 듣기','스포츠','슬로 모션','슬로우모션','시사','신나는','신비','어린이','여행','역사',
            '영화','오케스트라','웃기는','일렉트로닉','재즈','저속 촬영','전자','카페','컨트리',
            '코메디','클래식','타임랩스','팝','패션','펑키','펑키 재즈','평화로운','포크','희망','힙합']

search_song_indices = [model.wv.vocab[str(x)].index for x in audio_meta.keys() if str(x) in model.wv.vocab]
tag_indices = [model.wv.vocab[str(x)].index for x in tag_set if str(x) in model.wv.vocab]

audio_list = F.multiquery_retrieval(model.wv, model_input, search_song_indices)
tag_list = F.multiquery_retrieval(model.wv, model_input, tag_indices)

## your need to change audio_path based on your development setting
audio_path = "http://127.0.0.1:5000/static/audio_meta/audio/"
print([os.path.join(audio_path, fname) for fname in audio_list])

print(tag_list)
```

```
['있느냐”고 비꼬았다', '미래통합당 하태경 의원이', '매뉴얼이라도 있느냐”고 비꼬았다', '부동산', '미래통합당 하태경', '하태경 의원이', '아파트', '청와대', '“청와대에', '비꼬았다']
['부동산', '의원', '매뉴얼', '당', '청와대', '미래', '아파트', '하태경', '통합']
["http://127.0.0.1:5000/static/audio_meta/audio/25_Far_away.wav","http://127.0.0.1:5000/static/audio_meta/audio/65_Running_To_The_Sky.mp3","http://127.0.0.1:5000/static/audio_meta/audio/31_Lost_in_the_fog.wav","http://127.0.0.1:5000/static/audio_meta/audio/86_Ranking_show.mp3","http://127.0.0.1:5000/static/audio_meta/audio/66_Run.mp3"]
["블로그","뉴스","광고","광고하는","범죄"]
```


## Flask example
https://github.com/Dohppak/SBA_API_Flask