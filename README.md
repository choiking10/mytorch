# mytorch

[![Build Status](https://travis-ci.com/choiking10/mytorch.svg?branch=main)](https://travis-ci.com/choiking10/mytorch)

implementation of DeZero (deep learning from scratch-3)


## notebook 실행
```console
docker build . -t mytorch
docker run --gpus all -it --rm -v $PWD:/workspace -p 8001:8001 mytorch \
           jupyter notebook --allow-root --ip 0.0.0.0 --port 8001
```

## Dezero 외의 추가 구현한 부분
 - [x] 각종 테스트 코드 추가  
 - [x] CI 툴을 이용한 빌드 및 테스트 자동화  
 - [x] heap q 를 이용한 generation 정렬  


## Blogging 거리
 - [ ] 쉽게 내가 짠 backward 함수를 검증해볼 수 있을까? 
   - numerical_gradient_check를 활용한 접근법
 - [ ] 텐서 사용 시의 역전파 방법에 따른 계산 효율 (자동미분 forward 모드와 reverse 모드)
 - [ ] broadcast 함수의 역전파는 어떻게 이루어질까?
 - [ ] 행렬의 곱연산의 미분을 해보자.  
 - [ ] 왜 직접 구현해서 사용하는 것보다, pytorch에서 제공해주는 모듈을 쓰는게 좋을까? (메모리 관점에서의 접근)
   - step43의 내용 활용해서 블로그하면 좋을 듯
 - [ ] Optimizer와 관련된 내용도 포스팅할 만 할듯 
   - 왜 optimizer도 save해야 하는가?
   - Optimizer의 작동원리 등등

## 고지별 (4/5)
  - [고지 1](https://github.com/choiking10/mytorch/tree/chapter1)
  - [고지 2](https://github.com/choiking10/mytorch/tree/chapter2)
  - [고지 3 - 미완](https://github.com/choiking10/mytorch/tree/chapter3-incompletion)
  - [고지 4](https://github.com/choiking10/mytorch/tree/chapter4)

## Dezero 구현하기 진행도 Step 별 (52/60)
  - [step01](https://github.com/choiking10/mytorch/tree/step01)
  - [step02](https://github.com/choiking10/mytorch/tree/step02)
  - [step03](https://github.com/choiking10/mytorch/tree/step03)
  - [step04](https://github.com/choiking10/mytorch/tree/step04)
  - [step06](https://github.com/choiking10/mytorch/tree/step06)
  - [step07](https://github.com/choiking10/mytorch/tree/step07)
  - [step08](https://github.com/choiking10/mytorch/tree/step08)
  - [step09](https://github.com/choiking10/mytorch/tree/step09)
  - [step10](https://github.com/choiking10/mytorch/tree/step10)
  - [step11](https://github.com/choiking10/mytorch/tree/step11)
  - [step12](https://github.com/choiking10/mytorch/tree/step12)
  - [step13](https://github.com/choiking10/mytorch/tree/step13)
  - [step14](https://github.com/choiking10/mytorch/tree/step14)
  - [step16](https://github.com/choiking10/mytorch/tree/step16)
  - [step17](https://github.com/choiking10/mytorch/tree/step17)
  - [step18](https://github.com/choiking10/mytorch/tree/step18)
  - [step19](https://github.com/choiking10/mytorch/tree/step19)
  - [step21](https://github.com/choiking10/mytorch/tree/step21)
  - [step22](https://github.com/choiking10/mytorch/tree/step22)
  - [step23](https://github.com/choiking10/mytorch/tree/step23)
  - [step24](https://github.com/choiking10/mytorch/tree/step24)
  - [step26](https://github.com/choiking10/mytorch/tree/step26)
  - [step27](https://github.com/choiking10/mytorch/tree/step27)
  - [step28](https://github.com/choiking10/mytorch/tree/step28)
  - [step37 - 텐서를 다루다](https://github.com/choiking10/mytorch/tree/step37)
  - [step38 - 형상 변환 함수 (고차미분 적용)](https://github.com/choiking10/mytorch/tree/step38)
  - [step40 - 브로드캐스트 함수](https://github.com/choiking10/mytorch/tree/step40)
  - [step39 - 합계(Sum) 함수](https://github.com/choiking10/mytorch/tree/step39)
  - [stag41 - 행렬의 곱](https://github.com/choiking10/mytorch/tree/step41)
  - [step42 - 선형회귀](https://github.com/choiking10/mytorch/tree/step42)
  - [step43 - 신경망](https://github.com/choiking10/mytorch/tree/step43)
  - [step44 - 매개변수를 모아두는 계층](https://github.com/choiking10/mytorch/tree/step44)
  - [step45 - 계층를 모아두는 계층](https://github.com/choiking10/mytorch/tree/step45)
  - [step46 - Optimizer로 수행하는 매개변수 갱신](https://github.com/choiking10/mytorch/tree/step46)
  - [step47 - 소프트맥스 함수와 교차 엔트로피 오차](https://github.com/choiking10/mytorch/tree/step47)
  - [step48 - 다중 클래스 분류](https://github.com/choiking10/mytorch/tree/step48)
    - 데이터 시각화 코드 추가
  - [step49 - Dataset 클래스와 전처리](https://github.com/choiking10/mytorch/tree/step49)
  - [step50 - 미니배치를 뽑아주는 DataLoader](https://github.com/choiking10/mytorch/tree/step50)
  - [step51 - MNIST 학습](https://github.com/choiking10/mytorch/tree/step51)
  - [step52 - GPU 지원](https://github.com/choiking10/mytorch/tree/step52)
