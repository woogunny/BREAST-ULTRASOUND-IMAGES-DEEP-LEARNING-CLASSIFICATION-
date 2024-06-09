# AI1 프로젝트

##### 안녕하세요👋😀 Hi
![포켓몬 이미지](https://github.com/woogunny/BREAST-ULTRASOUND-IMAGES-DEEP-LEARNING-CLASSIFICATION-/blob/main/pocket.png)


# **프로젝트 제목 Project title** 

## **BREAST-ULTRASOUND-IMAGES-DEEP-LEARNING-CLASSIFICATION**

## ** 유방암 초음파 영상 딥러닝 분류 **


*******


# 프로젝트 개요 Project Overview

## <u>조기 발견은 조기 사망자 수를 줄이는 데 도움이 됩니다.</u>

## <u>Early detection helps to reduce the number of premature deaths.</u>
 


![example](https://github.com/woogunny/BREAST-ULTRASOUND-IMAGES-DEEP-LEARNING-CLASSIFICATION-/blob/main/example.png)


***********
# + 필요한 라이브러리(버전) 또는 프로그램 목록
# + List of required libraries (version) or programs


  + numpy                            1.25.2


  + pandas                           2.0.3


  + torch                            2.3.0+cu121


  + tensorflow                       2.15.0


  + scikit-learn                     1.2.2


  + matplotlib                       3.7.1


  + opencv-python                    4.8.0.76


*********


# 추후 개선 사항 : 추가로 개선해야할 내용에 대해 정리 또는 프로젝트 한계점 설명
# Future improvements: Organize additional improvements or explain project limitations
1. ***CycleGAN(2017)을 이용한 정량적 분석 통한 의료 데이터 불균형 해소 Resolving Medical Data Imbalance Through Quantitative Analysis Using CycleGAN (2017)****
2. ***성능 높히기 Increasing performance***


# <딥러닝>

# 데이터 세트 : 데이터셋에 대한 설명 및 출처
# Dataset: Description and Source of Dataset

## Breast Ultrasound Images Dataset in kaggle

>https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset


## 모델 설명 : 사용한 딥러닝 모델의 종류 및 선택 이유
## Model Description: Types of machine deep learning models used and reasons for selection

1. Deep-learning Model : SimpleCNN
+ 선택한 이유 : 처음에 무엇을 쓸지 몰라 간단한 실험용으로 먼저 써봤다.
+ Why I chose: I didn't know what to write at first, so I wrote it first for a simple experiment.


2. Deep-learning Model : VGG19
+ 선택한 이유 : 딥러닝 모델의 층을 깊게 쌓는 것을 시도하려고 써봤다.
+ Why I chose: I wrote it to try to build deep layers of deep learning models.


3. Deep-learning Model : ResNet18
+ 선택한 이유 : 모델 사용 처음에 SimpleCNN을 썼다 과적합과 기울기 소실 현상으로 문제를 겪다가 여러 (SimpleCNN, VGG19 처럼) ResNet18이 이러한 고질적인 문제를 해결해준다는 것을 알게 되었다.
+ Why we chose: We used SimpleCNN at the beginning of using the model. After experiencing problems with overfitting and gradient loss, we find that ResNet18 (like SimpleCNN, VGG19) solves these chronic problems.


# 실험 결과 : 모델 평가에 사용된 지표와 결과(표 또는 그래프)
# Experimental Results: Indicators and results (table or graph) used for model evaluation

+ First Ablation Study
![example1](https://github.com/woogunny/BREAST-ULTRASOUND-IMAGES-DEEP-LEARNING-CLASSIFICATION-/blob/main/example1.png)

+ Second Ablation Study (After solving the data balance)
![example2](https://github.com/woogunny/BREAST-ULTRASOUND-IMAGES-DEEP-LEARNING-CLASSIFICATION-/blob/main/example.png)





