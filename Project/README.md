# Final Project - Hand Written Letter Image Sorting

10장의 알파벳 필기 이미지를 입력 받아, 정렬된 순서로 class를 차례로 예측​

Image: single channel의 gray-scale의 알파벳 필기​

Label: A(a) ~ Z(z) 까지의 class index (알파벳 순으로 sorted)​

Example
- R, h, t, b, h, w, Z, M, t, C -> 1, 2, 7, 7, 12, 17, 19, 19, 22, 25

![image](https://user-images.githubusercontent.com/37990408/230401887-50ccd203-e7c2-4de6-8d48-7dd529934be3.png)

## Architecture

### Shared CNN + RNN
- 10 장의 이미지를 동일 CNN에 입력
- 출력된 10개의 hidden representation을 RNN의 input으로 사용
- 11~20번째 RNN output은 class 개수에 해당하는 logit (NOTE: do not normalize at models.py)

![image](https://user-images.githubusercontent.com/37990408/230403258-c8902db5-a050-4d89-877b-06f79a5485f5.png)

## Search Space

### Constraints
- 주어진 학습 데이터 외의 추가 데이터 사용 금지
- Pre-trained model 활용 금지
- 1D/2D convolution for CNN
- Self-attention 등 CNN과 RNN 범위를 벗어난 모듈 사용 금지

### CNN + RNN 구조와 input/output의 shape만 유지한다면 자유롭게 수정 가능
- Data augmentation
- Teacher forcing
- Hyperparameter search 등등
- NOTE: 별도 패키지 설치는 불가능

## Project Structure

## modules.py
### CustomCNN
- 10장의 image 입력에 사용되는 모듈

```__init__```
- Parameter 등 변수 자유롭게 선언 가능
- Hyperparameter를 입력 받기 위한 argument도 자유롭게 지정 가능

```forward```
- Inputs: (batch * seq_len, 1, H, W)
- Outputs: (seq_len, batch, hidden_dim)

### LSTM
- CustomCNN의 output을 입력 받고, 10개의 class prediction을 출력하는 모듈

```__init__```
- Parameter 등 변수 자유롭게 선언 가능
- Hyperparameter를 입력 받기 위한 argument도 자유롭게 지정 가능
- LSTM 모듈을 사용하였으나, GRU 등으로 대체 가능
- RNN input dim과 output dim 고려

```forward```
- feature: CNN의 output / 이전 step의 output/ 이전 step의 ground truth embedding
- h, c: LSTM의 previous hidden state와 cell state
- output: feature의 sequence length 만큼의 RNN output
- h_next, c_next: current hidden state와 cell state

## models. py
CustomCNN과 LSTM을 이용해 class prediction 수행

```__init__```
- 여러 hyperparameters를 argument로 받음
- 자유롭게 수정 및 추가 가능 NOTE: sequence_length, num_classes는 고정
- CustomCNN, LSTM을 선언
- 추가 부수적인 parameter 선언 가능

```Forward```
- CustomCNN, LSTM (, label 정보)를 사용하여 10개의 위치에 대한 확률 분포를 출력
- Teacher-forcing: training에 한하여 RNN에 ground truth class를 입력하는 방법 (optional)
- Inputs: (images, labels) or images
- hidden/cell_state: initial state로, zero tensor
- Outputs: (batch*10, 26)의 logit predictionNOTE: softmax를 취하지 않고 출력할 것NOTE: shape 준수할 것

```Teacher-Forcing```
- Sequence를 순차적으로 예측하는 task에서, 학습 시 이전 step의 ground truth를 입력해주는 방법
- ConvLSTM의 forward 함수에서 주어지는 ‘labels’ 값을 사용
- 사용할지 말지, 어떤 representation을 넣을지, 어느 비율로 사용할지, 어떻게 scheduling할지 등은 자유
- Example: R, h, t, b, h, w, Z, M, C, t  1, 2, 7, 7, 12, 17, 19, 19, 22, 25

## data_utils.py
### Dataloader에 필요한 Dataset 정의
- Mydataset
- Mytensordataset
### Collate_fn: batch를 전처리하는 코드
- Data augmentation 등 수행 가능
- Return하는 img, label의 shape 등은 바꾸지 말 것

## main.ipynb
### Dataloader, model 등을 선언하여 학습 및 평가
- 자세한 내용은 notebook의 markdown 참조

```순서```
- 패키지 import
- (optional) 실제 샘플 visualization
- Dataloader 선언
- Hyperparameters, 모델, optimizer 등 선언
- 학습
- Checkpoint load 및 평가
- (optional) ensemble learning 및 평가

### Ensemble
- Pytorch-Ensemble (customized)을 통한 ensemble learning
- Voting, Bagging, Fusion, GradientBoosting, SoftGradientBoosting 중 택 1
- pip install git+https://github.com/snuml2021tmp/Ensemble-Pytorch.git
- Ensemble type, optimizer, learning rate scheduler를 set할 수 있음











