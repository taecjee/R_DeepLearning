## 컨볼루션 계산
a <- matrix(c(0.5, 0.3, 0.1, 0.2, 0.6, 0.1, 0.1, 0.1, 0.7), nrow = 3, byrow = TRUE)
b <- matrix(c(0.5, 0.6, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0), nrow = 3, byrow = TRUE)

conv <- matrix(c(3, 1, 1, 1, 3, 1, 1, 1, 3), nrow = 3)

a * conv
sum(a*conv)

b * conv
sum(b * conv)

## MNIST 데이터 다운로드
dataDirectory <- "../data"
if (!file.exists(paste(dataDirectory,'/train.csv',sep="")))
{
  link <- 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/mnist_csv.zip'
  if (!file.exists(paste(dataDirectory,'/mnist_csv.zip',sep="")))
    download.file(link, destfile = paste(dataDirectory,'/mnist_csv.zip',sep=""))
  unzip(paste(dataDirectory,'/mnist_csv.zip',sep=""), exdir = dataDirectory)
  if (file.exists(paste(dataDirectory,'/test.csv',sep="")))
    file.remove(paste(dataDirectory,'/test.csv',sep=""))
}

## 데이터 읽기
train <- read.csv(paste(dataDirectory,'/train.csv',sep=""), header=TRUE, nrows=20)

tail(train[,1:6])
tail(train[,(ncol(train)-5):ncol(train)])

## 데이터 확인
plotInstance <-function (row,title="")
{
  mat <- matrix(row,nrow=28,byrow=TRUE)
  mat <- t(apply(mat, 2, rev))
  image(mat, main = title,axes = FALSE, col = grey(seq(0, 1, length = 256)))
}
par(mfrow = c(3, 3))
par(mar=c(2,2,2,2))
for (i in 1:9)
{
  row <- as.numeric(train[i,2:ncol(train)])
  plotInstance(row, paste("index:",i,", label =",train[i,1]))
}
par(mfrow = c(1, 1))

## 데이터 분리
require(mxnet)
options(scipen=999)

dfMnist <- read.csv("../data/train.csv", header=TRUE)
yvars <- dfMnist$label
dfMnist$label <- NULL

set.seed(42)
train <- sample(nrow(dfMnist),0.9*nrow(dfMnist))
test <- setdiff(seq_len(nrow(dfMnist)),train)
train.y <- yvars[train]
test.y <- yvars[test]
train <- data.matrix(dfMnist[train,])
test <- data.matrix(dfMnist[test,])

rm(dfMnist,yvars)

## 데이터 변환
train <- t(train / 255.0)
test <- t(test / 255.0)

require(ggplot2)
table(train.y)
ggplot(data.frame(train.y), aes(x=train.y)) + geom_bar()

table(test.y)
ggplot(data.frame(test.y), aes(x=test.y)) + geom_bar()

## 표준 신경망 모델 설계
data <- mx.symbol.Variable("data")
fullconnect1 <- mx.symbol.FullyConnected(data, name="fullconnect1", num_hidden=256)
activation1  <- mx.symbol.Activation(fullconnect1, name="activation1", act_type="relu")
fullconnect2 <- mx.symbol.FullyConnected(activation1, name="fullconnect2", num_hidden=128)
activation2  <- mx.symbol.Activation(fullconnect2, name="activation2", act_type="relu")
fullconnect3 <- mx.symbol.FullyConnected(activation2, name="fullconnect3", num_hidden=10)
softmax      <- mx.symbol.SoftmaxOutput(fullconnect3, name="softmax")

## 표준 신경망 모델 학습
devices <- mx.cpu()
mx.set.seed(0)
model <- mx.model.FeedForward.create(softmax, X=train, y=train.y,
                                     ctx=devices,array.batch.size=128,
                                     num.round=10,
                                     learning.rate=0.05, momentum=0.9,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(1))

## 표준 신경망 모델 평가
preds1.tr <- predict(model, train)
pred.label1.tr <- max.col(t(preds1.tr)) - 1
res1.tr <- data.frame(cbind(train.y,pred.label1.tr))
table(res1.tr)
(accuracy1.tr <- sum(res1.tr$train.y == res1.tr$pred.label1.tr) / nrow(res1.tr))

preds1 <- predict(model, test)
pred.label1 <- max.col(t(preds1)) - 1
res1 <- data.frame(cbind(test.y,pred.label1))
table(res1)
accuracy1 <- sum(res1$test.y == res1$pred.label1) / nrow(res1)
accuracy1

## LeNet 모델 설계
data <- mx.symbol.Variable('data')
# first convolution layer
convolution1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=64)
activation1  <- mx.symbol.Activation(data=convolution1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=activation1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# second convolution layer
convolution2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=32)
activation2  <- mx.symbol.Activation(data=convolution2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=activation2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))

# flatten layer and then fully connected layers
flatten <- mx.symbol.Flatten(data=pool2)
fullconnect1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=512)
activation3  <- mx.symbol.Activation(data=fullconnect1, act_type="relu")
fullconnect2 <- mx.symbol.FullyConnected(data=activation3, num_hidden=10)
# final softmax layer
softmax <- mx.symbol.SoftmaxOutput(data=fullconnect2)

## 데이터 형태 변경
train.array <- train
dim(train.array) <- c(28,28,1,ncol(train))
test.array <- test
dim(test.array) <- c(28,28,1,ncol(test))

## LeNet 모델 학습
devices <- mx.cpu()
#devices <- mx.gpu()
mx.set.seed(0)
logger <- mx.metric.logger$new()
model2 <- mx.model.FeedForward.create(softmax, X=train.array, y=train.y,
                                      ctx=devices,array.batch.size=128,
                                      num.round=10,
                                      learning.rate=0.05, momentum=0.9, wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      eval.data=list(data=test.array,labels=test.y),
                                      epoch.end.callback=mx.callback.log.train.metric(100,logger))

## LeNet 모델 평가
preds2.tr <- predict(model2, train.array)
pred.label2.tr <- max.col(t(preds2.tr)) - 1
res2.tr <- data.frame(cbind(train.y,pred.label2.tr))
table(res2.tr)
(accuracy2.tr <- sum(res2.tr$train.y == res2.tr$pred.label2.tr) / nrow(res2.tr))

preds2 <- predict(model2, test.array)
pred.label2 <- max.col(t(preds2)) - 1
res2 <- data.frame(cbind(test.y,pred.label2))
table(res2)
accuracy2 <- sum(res2$test.y == res2$pred.label2) / nrow(res2)
accuracy2

## LeNet 모델 보기
graph.viz(model$symbol, type = "vis")
graph.viz(model2$symbol)

## LeNet 모델 학습 과정 보기
dfLogger<-as.data.frame(round(logger$train,3))
dfLogger2<-as.data.frame(round(logger$eval,3))
dfLogger$eval<-dfLogger2[,1]
colnames(dfLogger)<-c("train","eval")
dfLogger$epoch<-as.numeric(row.names(dfLogger))

data_long <- reshape2::melt(dfLogger, id="epoch") 

ggplot(data=data_long,
       aes(x=epoch, y=value, colour=variable,label=value)) +
  ggtitle("Model Accuracy") +
  ylab("accuracy") +
  geom_line()+geom_point() +
  geom_text(aes(label=value),size=3,hjust=0, vjust=1) +
  theme(legend.title=element_blank()) +
  theme(plot.title=element_text(hjust=0.5)) +
  scale_x_discrete(limits= 1:nrow(dfLogger))
























######################################
## 연습문제
######################################
## 데이터 읽기
source("ChapterIII/exploreFashion.R")

load_mnist()
show_digit(train$x[5,])

head(train$x[,1:5])
head(train$x[,(ncol(train$x)-5):ncol(train$x)])

## 데이터 확인 하기
plotInstance <-function (row,title="")
{
  mat <- matrix(row,nrow=28,byrow=TRUE)
  mat <- t(apply(mat, 2, rev))
  image(mat, main = title,axes = FALSE, col = grey(seq(0, 1, length = 256)))
}
labels<-c("T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")

par(mfrow = c(4, 4))
par(mar=c(2,2,2,2))
for (i in 1:16)
{
  row <- as.numeric(train$x[i,])
  num_label<-train$y[i]
  plotInstance(row, paste("index:",i,", label =",labels[num_label+1]))
}
par(mfrow = c(1, 1))

## 데이터 형태 변경
train$x <- t(train$x / 255.0)
test$x <- t(test$x / 255.0)

train.array <- train$x
dim(train.array) <- c(28, 28, 1, ncol(train$x))
test.array <- test$x
dim(test.array) <- c(28, 28, 1, ncol(test$x))

table(train$y)

act_type1="relu"
devices <- mx.cpu()
mx.set.seed(0)

## 모델 설계
data <- mx.symbol.Variable('data')
# first convolution layer
convolution1 <- mx.symbol.Convolution(data=data, kernel=c(5,5),
                                      stride=c(1,1), pad=c(2,2), num_filter=64)
activation1  <- mx.symbol.Activation(data=convolution1, act_type=act_type1)
pool1        <- mx.symbol.Pooling(data=activation1, pool_type="max",
                                  kernel=c(2,2), stride=c(2,2))

# second convolution layer
convolution2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5),
                                      stride=c(1,1), pad=c(2,2), num_filter=32)
activation2  <- mx.symbol.Activation(data=convolution2, act_type=act_type1)
pool2        <- mx.symbol.Pooling(data=activation2, pool_type="max",
                                  kernel=c(2,2), stride=c(2,2))

# flatten layer and then fully connected layers with activation and dropout
flatten <- mx.symbol.Flatten(data=pool2)
fullconnect1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=512)
activation3 <- mx.symbol.Activation(data=fullconnect1, act_type=act_type1)
drop1 <- mx.symbol.Dropout(data=activation3,p=0.4)
fullconnect2 <- mx.symbol.FullyConnected(data=drop1, num_hidden=10)
# final softmax layer
softmax <- mx.symbol.SoftmaxOutput(data=fullconnect2)

## 모델 학습
model2 <- mx.model.FeedForward.create(softmax, X=train.array, y=train$y,
                                      ctx=devices, num.round=20,
                                      array.batch.size=64,
                                      learning.rate=0.05, momentum=0.9,
                                      wd=0.00001,
                                      eval.metric=mx.metric.accuracy,
                                      eval.data=list(data=test.array,labels=test$y),
                                      epoch.end.callback=mx.callback.log.train.metric(1))

## 모델 평가
preds2 <- predict(model2, test.array)
pred.label2 <- max.col(t(preds2)) - 1
res2 <- data.frame(cbind(test$y,pred.label2))
table(res2)
accuracy2 <- sum(res2$test$y == res2$pred.label2) / nrow(res2)
accuracy2