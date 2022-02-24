# proj5-spam-sms-classifier
RNN, ELMo, Xgboost등의 머신러닝 모델들을 이용한 스팸 문자 분류 및 분석


## 📂 File Structure
```
├── README.md
├── data
│   ├── spam.csv
│   ├── sms_data.csv
└── docs
│   ├── keynote.pdf
│   └── keynote.keynote
└── Preprocessing
│   ├── 1_Preprocessing.ipynb
│   ├── 1_Preprocessing.py
│   ├── spam_prc.csv
│   └── Visualization
│       ├── all.png
│       ├── barchart.png
│       ├── lengthchart.png
│       ├── spam.png
│       └── ham.png
└── Vanila-Rnn
│   ├── 2_Vanila_RNN.ipynb
│   ├── 2_Vanila_rnn.py
│   ├── spam_prc.csv
│   ├── CheckPoint
│   ├── Model
│   └── Visualization
└── Ensemble
│   ├── 3_ML_modeling.ipynb
│   ├── 3_ML_modeling.py
│   ├── sms_data.csv
│   └── Visualization
└── ELMO
│   ├── 4_Elmo_Model.ipynb
│   ├── 4_elmo_model.py
│   ├── sms_data.csv
│   ├── Model
│   └── Visualization.py
```

## 📦 SMS Spam Collection Dataset
> from UCI Machine Learning (5,572 data sample)

```
- v1 : spam, ham category
- v2 : text
- unnamed 2
- unnamed 3
- unnmaed 4
```
- [download](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

# 1. Intro

> 스팸과 스팸이 아닌 문자를 확실하면서 빠르게 구별하여 제거가 용이하도록 만들기 위한 배경

스팸 문자를 어떤 모델이 확실하게 분류 할 수 있을까를 중점으로 실험적인 프로젝트를 준비하였습니다.

[v] 다양한 모델을 활용하여 성능을 높일 수 있지 않을까?

[v] 딥러닝 뿐만 아니라 머신러닝을 이용하면 어떤 모델이 더 좋은 성능을 낼까?

# 2. Data Preprocessing
### 기본 전처리
- `isnull().value.any()` : null 값이 있는 데이터를 확인하고 제거한다.
- `.nunique()` : 중복이 있는 데이터를 제거한다.
- `columns modify` : 학습을 효울적으로 진행하기 위해 `spam`과 `ham` 을 0과 과 변경한다.
### 토큰화
- 기존 데이터셋에서 타겟 데이터와 학습 데이터를 나눈다.
- 학습할 데이터는 토큰화를 진행하여 빈도에따라 정수로 인덱싱하고 패딩처리를 진행하여 훈련데이터와 시험 데이터로 나누게 된다.

### 시각화

- 문자의 최대 길이와 이에 따른 데이터 수 시각화 ( 대체적으로 길이가 50 이하라는 것을 알 수 있다.)
- 빈도수를 보게되면 주로 스팸 메세지는 150정도의 길이며, 아닌 일반 메세지는 50정도 길이가 가장 많음을 알 수 있었다.
- 비스팸의 경우 150 글자로 문자의 길이의 분포가 다양하게 나타나지만, 스팸의 경우 170글자로 문자의 길이의 분포를 보였다
- 워드 클라우드로 나타냈을 때, 전체적으로는 `now` 나 `will` 단어들이 눈에 띄었다.
- 스팸과 일반 문자들을 비교했을 때, 스팸 문자 데이터는 `free` 나 `now` `mobile` `text` 등 사용자에게 무언가를 요구하거나 광고성 단어들이 빈도수가 많았고 이외로 일반 문자 데이터는 일상적으로 사용하는 단어들이 빈도수가 많았습니다.

# 3. Modeling
### RNN
RNN 모델은 딥러닝에서 가장 기본적인 시퀀스 모델입니다. \
이 모델을 사용한 이유는 단어 시퀀스에 대해 Many-to-one으로 하는 스팸메일 분류나 감성 분류를 진행할 수 있기 때문입니다.


모델링을 위해 에폭 수를 10, …. 임베딩 벡터의 차원을 32로 두어 학습을 진행하였습니다.\
콜백함수로 earlystopping과 checkpoint를 사용하여 학습 동안 최고의 모델을 저장하였습니다.

주로 **2번째 학습부터 손실값이 감소**를 했고 **정확도는 0.9816**이라는 가장 높은 정확도가 나왔습니다.

### Ensemble
이번에는 머신러닝 모델을 이용하여 스팸 문자를 분류해보았습니다.\
estimator와 max_features를 하이퍼파라미터로 하여 최적의 결과를 나타낸 결과 가장 정확도가 높은 모델은 XGBoost가 나왔습니다. 

### ELMo
엘모 모델은 약자로 임베딩 프롬 랭귀지 모델입니다.\
즉, 언어모델로 임베딩하는 것입니다. 이 엘모의 가장 큰 특징은 사전 훈련 모델이라 tensorflow Hub를 통해 구현할 수 있었습니다.

> 양방향 RNN과 언뜻보면 비슷해보이지만 전자와 다르게 ELMo의 biLM(비아이엘엠)은 서로 연결하여 다음층 입력으로 사용하지 않고 두개의 언어모델을 별개의 모델로보고 학습한 다는 것입니다. 

3번의 학습을 통해 진행 한 결과, 그 중 **Epoch이 4일 때 가장 낮은 손실**이 나왔으며 **가장 높은 정확도는 0.993** 이었습니다. 한 번 학습 의 소요한 시간은 약 3시간 정도 였습니다.

# 4. Outtro
### Overfitting
이번 프로젝트를 진행하면서 결과적으로 데이터셋 크기가 작아 에폭수가 3~5만 넘어가도 과적합이 되는 경우가 많았다.\
 그래서 에폭수를 늘려 긴 학습을 진행했다.

### ML AND DL
그리고  ELMo를 기반으로 일반 RNN 그리고 다양한 머신러닝 앙상블 모델을 활용하여 학습을 진행하였는데 그럼에도 문맥을 잘 고려할 수 있는 임베딩 모델이 가장 높은 정확도를 보였던 것 같다.

### YOUTUBE ▶️
https://www.youtube.com/watch?v=Z_UMSz2yWwY

### Reference
- https://www.kaggle.com/faressayah/natural-language-processing-nlp-for-beginners
- https://wikidocs.net/22886
- https://www.kaggle.com/andreshg/nlp-glove-bert-tf-idf-lstm-explained
- https://velog.io/@changhtun1/ensemble