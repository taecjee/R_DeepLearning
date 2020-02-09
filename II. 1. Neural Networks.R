## MNIST 데이터
dataDirectory <- "../data"
if (!file.exists(paste(dataDirectory, '/train.csv', sep=""))) {
  link <- 'https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/data/mnist_csv.zip'
  if (!file.exists(paste(dataDirectory, '/mnist_csv.zip', sep=""))) 
    download.file(link, destfile = paste(dataDirectory, '/mnist_csv.zip', sep=""))
  unzip(paste(dataDirectory, '/mnist_csv.zip', sep = ""), exdir=dataDirectory)
}

mnist.train <- read.csv(paste(dataDirectory, '/train.csv', sep=""))
mnist.test <- read.csv(paste(dataDirectory, '/test.csv', sep=""))

dim(mnist.train)
dim(mnist.test)

head(colnames(mnist.train), 4)
tail(colnames(mnist.train), 4)
head(mnist.train[,1:4])

## MNIST 데이터 시각화
im <- matrix((mnist.train[4,2:ncol(mnist.train)]), nrow=28, ncol=28)
im_numbers <- apply(im, 2, as.numeric)
image(1:28, 1:28, im_numbers, col=gray((0:255)/255))

## MNIST 데이터 시각화 2
im <- matrix(nrow=28, ncol=28)
j <- 1
for(i in 28:1){
  im[,i] <- (as.numeric(mnist.train[4,2:ncol(mnist.train)]))[j:(j+27)]
  j <- j+28
} 
im_numbers <- apply(im, 2, as.numeric)
image(1:28, 1:28, im_numbers, col=gray((255:0)/255))

## MNIST 데이터 시각화 함수
img <- function(d, row_index){
  # 행을 숫자 벡터로 변경
  r <- as.numeric(d[row_index, 2:785])
  # 행렬 선언 및 데이터 적재
  im <- matrix(nrow = 28, ncol = 28)
  j <- 1
  for(i in 28:1){
    im[,i] <- r[j:(j+27)]
    j <- j+28
  }  
  # 제목이 있는 이미지 출력
  image(x = 1:28, y = 1:28, z = im, col=gray((255:0)/255), main = paste("Number:", d[row_index, 1]))
}

img(mnist.train, 4)

## 여러 이미지 시각화
par(mfcol = c(6, 6))
par(mar = c(1, 0, 3, 0), xaxs='i', yaxs='i')
for (idx in 1:36) { 
  img(mnist.train, idx)
}
par(mfcol = c(1, 1))
par(mar = c(5.1, 4.1, 4.1, 2.1), xaxs='r', yaxs='r')

## 문제 단순화
library(dplyr)
digits_nn <- mnist.train[(mnist.train$label == 5) | (mnist.train$label == 6),]
digits_nn <- mnist.train %>% filter(label %in% c(5, 6))
digits_nn$y <- digits_nn$label
digits_nn$label <- NULL
table(digits_nn$y)

digits_nn$y <- ifelse(digits_nn$y == 5, 0, 1)
table(digits_nn$y)

set.seed(42)
sample <- sample(nrow(digits_nn), nrow(digits_nn) * 0.8)
test <- setdiff(seq_len(nrow(digits_nn)), sample)

## 열의 값이 모두 같은 경우 제거
var(digits_nn[sample, 1], na.rm = TRUE)
digits.X2 <- digits_nn[, apply(digits_nn[sample, 1:(ncol(digits_nn)-1)], 2, var, na.rm=TRUE) != 0]
length(digits.X2)
length(digits_nn)

## PCA 적용
tic <- proc.time()
df.pca <- prcomp(digits.X2[sample,], center = TRUE, scale. = TRUE)
proc.time() - tic
#    user  system elapsed 
#    9.10    0.05    9.16 
s <- summary(df.pca)
cumprop <- s$importance[3, ]
plot(cumprop, type = "l", main = "Cumulative sum", xlab = "PCA component")
# this plot shows that we can use the first 100 variable (the principal components) and this will account for over 80% of the variance in the original data

# selects out the principal components that account for 50% of our variance
num_cols <- min(which(cumprop > 0.5))
cumprop[num_cols]


## neuralnet 신경망 모델 생성
#install.packages("neuralnet")
library(neuralnet)
newdat <- data.frame(df.pca$x[, 1:num_cols])
newdat$y <- digits_nn[sample, "y"]
col_names <- names(newdat)
f <- as.formula(paste("y ~", paste(col_names[!col_names %in% "y"], collapse="+")))
tic <- proc.time()
nn <- neuralnet(f, data=newdat, hidden=c(4,2), linear.output = FALSE)
proc.time() - tic
#    user  system elapsed 
# 110.42    0.07  110.61 

plot(nn)

## neuralnet 신경망 모델 평가
test.data <- predict(df.pca, newdata = digits_nn[test, colnames(digits.X2)])
test.data <- as.data.frame(test.data)
preds <- compute(nn, test.data[, 1:num_cols])
preds <- ifelse(preds$net.result > 0.5, "1", "0")
t <- table(digits_nn[test, "y"], preds, dnn = c("Actual", "Predicted"))
acc <- round(100.9 * sum(diag(t))/sum(t), 2)
print(t)
#       Predicted
# Actual   0   1
#      0 740  17
#      1  17 813
print(sprintf(" accuracy = %1.2f%%", acc))
# [1] " accuracy = 98.74%"

## 전체 변수를 이용한 neuralnet
tic <- proc.time()
nn.all <- neuralnet( y ~ ., data = digits_nn[sample,], hidden = c(4,2), linear.output = FALSE)
proc.time() - tic
#  user  system elapsed 
# 618.50    0.98   625.92 
plot(nn.all)
preds.all <- compute(nn.all, digits_nn[test, ])
preds.all <- ifelse(preds.all$net.result > 0.5, "1", "0")
t.all <- table(digits_nn[test, "y"], preds.all, dnn = c("Actual", "Predicted"))
acc.all <- round(100.9 * sum(diag(t.all))/sum(t.all), 2)
print(t.all)
#        Predicted
# Actual   0   1
#      0 731  26
#      1  34 796
print(sprintf(" accuracy = %1.2f%%", acc.all))
# [1] " accuracy = 97.09%"

## 다중 분류 데이터
sample.2 <- sample(nrow(mnist.train), 6000)
train.digits <- sample.2[1:5000]
test.digits <- sample.2[5001:6000]

digits.X <- mnist.train[train.digits, -1]
digits.y_n <- mnist.train[train.digits, 1]
mnist.train$label <- factor(mnist.train$label, levels = 0:9)
digits.y <- mnist.train[train.digits, 1]

digits.test.X <- mnist.train[test.digits, -1]
digits.test.y <- mnist.train[test.digits, 1]

barplot(table(digits.y), main = "Distribution of y values (train)")
barplot(table(digits.test.y), main = "Distribution of y values (test)")

## nnet을 이용한 데이터 분류 1
#install.packages("caret")
#install.packages("e1071")
library(nnet)
library(caret)
library(e1071)
set.seed(42)
tic <- proc.time()
digits.m1 <- caret::train(digits.X, digits.y,
                          method = "nnet",
                          tuneGrid = expand.grid(
                            .size = c(5),
                            .decay = c(0.1)),
                          trControl = trainControl(method = "none"),
                          MaxNWts = 10000,
                          maxit = 100)
print(proc.time() - tic)
#    user  system elapsed 
#    0.37    0.00    0.45 

## nnet을 이용한 데이터 예측 1
digits.yhat1 <- predict(digits.m1, newdata = digits.test.X)
accuracy <- 100.0 * sum(digits.yhat1 == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy))
# [1] " accuracy = 38.60%"
barplot(table(digits.yhat1), main = "Distribution of y values (model 1)")
table(digits.test.y, digits.yhat1)
caret::confusionMatrix(xtabs(~digits.yhat1 + digits.test.y))


## nnet을 이용한 데이터 분류 2
set.seed(42)
tic <- proc.time()
digits.m2 <- caret::train(digits.X, digits.y,
                          method = "nnet",
                          tuneGrid = expand.grid(
                            .size = c(10),
                            .decay = 0.1
                          ),
                          trControl = trainControl(method = "none"),
                          MaxNWts = 50000,
                          maxit = 100)
print(proc.time() - tic)
#    user  system elapsed 
#  148.00    0.02  148.24 

## nnet을 이용한 데이터 예측 2
digits.yhat2 <- predict(digits.m2, newdata = digits.test.X)
accuracy <- 100.0 * sum(digits.yhat2 == digits.test.y)/length(digits.test.y)
# [1] " accuracy = 72.20%"
print(sprintf(" accuracy = %1.2f%%", accuracy))
barplot(table(digits.yhat2), main = "Distribution of y values (model 2)")
table(digits.test.y, digits.yhat2)
caret::confusionMatrix(xtabs(~digits.yhat2 + digits.test.y))


## nnet을 이용한 데이터 분류 3
set.seed(42)
tic <- proc.time()
digits.m3 <- caret::train(digits.X, digits.y,
                          method = "nnet",
                          tuneGrid = expand.grid(
                            .size = c(40),
                            .decay = 0.1
                          ),
                          trControl = trainControl(method = "none"),
                          MaxNWts = 50000,
                          maxit = 100)
print(proc.time() - tic)
#    user  system elapsed 
# 2164.09    1.25 2168.64 

## nnet을 이용한 데이터 예측 3
digits.yhat3 <- predict(digits.m3, newdata = digits.test.X)
accuracy <- 100.0 * sum(digits.yhat3 == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy))
# [1] " accuracy = 83.70%"
barplot(table(digits.yhat3), main = "Distribution of y values (model 3)")
table(digits.test.y, digits.yhat3)
caret::confusionMatrix(xtabs(~digits.yhat3 + digits.test.y))


## RSNNS를 이용한 데이터 분류 (Stuttgart Neural Networks Simulator)
install.packages("RSNNS")
library(RSNNS)

# one-hot encoding
head(decodeClassLabels(digits.y))

set.seed(42)
tic <- proc.time()
digits.m4 <- mlp(as.matrix(digits.X),
                 decodeClassLabels(digits.y),
                 size = 40,
                 learnFunc = "Rprop",
                 shufflePatterns = FALSE,
                 maxit = 80)
print(proc.time() - tic)
#    user  system elapsed 
#   66.92    0.01   67.11


## RSNNS를 이용한 데이터 예측
digits.yhat4 <- predict(digits.m4, newdata = digits.test.X)
digits.yhat4 <- encodeClassLabels(digits.yhat4)
accuracy <- 100.0 * sum(I(digits.yhat4 - 1) == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy))
# [1] " accuracy = 81.00%"
barplot(table(digits.yhat4), main = "Distribution of y values (model 4)")
table(digits.test.y, digits.yhat4)

## 실환경에서 데이터 예측 방법 1
digits.yhat4_b <- predict(digits.m4, newdata = digits.test.X)
head(round(digits.yhat4_b, 2))

digits.yhat4_b_1 <- encodeClassLabels(digits.yhat4_b, method = "WTA", l = 0, h = 0)
table(digits.yhat4_b_1)
#   1   2   3   4   5   6   7   8   9  10 
# 117 109 106 110  96  76  88 107  79 112
accuracy_b_1 <- 100.0 * sum(I(digits.yhat4_b_1 - 1) == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy_b_1))
# [1] " accuracy = 81.00%"

## 실환경에서 데이터 예측 방법 2
digits.yhat4_b_2 <- encodeClassLabels(digits.yhat4_b, method = "WTA", l = 0, h = 0.5)
table(digits.yhat4_b_2)
#   0   1   2   3   4   5   6   7   8   9  10 
# 139 106 103  95  88  78  57  85  91  72  86 
accuracy_b_2 <- 100.0 * sum(I(digits.yhat4_b_2 - 1) == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy_b_2))
# [1] " accuracy = 75.40%"
unknown_b_2 <- 100.0 * length(which(digits.yhat4_b_2 == 0))/length(digits.yhat4_b_2)
print(sprintf(" unknown = %1.2f%%", unknown_b_2))
# [1] " unknown = 13.90%"
print(sprintf(" total accuracy = %1.2f%%", accuracy_b_2 + unknown_b_2))
# [1] " total accuracy = 89.30%"

## 실환경에서 데이터 예측 방법 3
digits.yhat4_b_3 <- encodeClassLabels(digits.yhat4_b, method = "WTA", l = 0.2, h = 0.5)
table(digits.yhat4_b_3)
#   0   1   2   3   4   5   6   7   8   9  10 
# 168 102 103  92  87  75  53  84  89  67  80 
accuracy_b_3 <- 100.0 * sum(I(digits.yhat4_b_3 - 1) == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy_b_3))
# [1] " accuracy = 74.00%"
unknown_b_3 <- 100.0 * length(which(digits.yhat4_b_3 == 0))/length(digits.yhat4_b_3)
print(sprintf(" unknown = %1.2f%%", unknown_b_3))
# [1] " unknown = 16.80%"
print(sprintf(" total accuracy = %1.2f%%", accuracy_b_3 + unknown_b_3))
# [1] " total accuracy = 90.80%"

## 실환경에서 데이터 예측 방법 4
digits.yhat4_b_4 <- encodeClassLabels(digits.yhat4_b, method = "402040", l = 0.4, h = 0.6)
table(digits.yhat4_b_4)
#   0   1   2   3   4   5   6   7   8   9  10 
# 260  94  99  82  77  66  43  81  78  55  65
accuracy_b_4 <- 100.0 * sum(I(digits.yhat4_b_4 - 1) == digits.test.y)/length(digits.test.y)
print(sprintf(" accuracy = %1.2f%%", accuracy_b_4))
# [1] " accuracy = 67.70%"
unknown_b_4 <- 100.0 * length(which(digits.yhat4_b_4 == 0))/length(digits.yhat4_b_4)
print(sprintf(" unknown = %1.2f%%", unknown_b_4))
# [1] " unknown = 26.00%"
print(sprintf(" total accuracy = %1.2f%%", accuracy_b_4 + unknown_b_4))
# [1] " total accuracy = 93.70%"

## 과적합 (Overfitting)
# 학습 데이터에 대한 인식률
digits.yhat4.train <- predict(digits.m4)
digits.yhat4.train <- encodeClassLabels(digits.yhat4.train)
accuracy <- 100.0*sum(I(digits.yhat4.train - 1)==digits.y)/length(digits.y)
print(sprintf(" accuracy = %1.2f%%",accuracy))

# 모든 모델의 학습 데이터에 대한 인식률
digits.yhat1.train <- predict(digits.m1)
digits.yhat2.train <- predict(digits.m2)
digits.yhat3.train <- predict(digits.m3)
digits.yhat4.train <- predict(digits.m4)
digits.yhat4.train <- encodeClassLabels(digits.yhat4.train)

# 학습 데이터 인식률과 테스트 데이터 인식률을 결합
measures <- c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")
n5.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat1.train))
n5.outsample <- caret::confusionMatrix(xtabs(~digits.test.y + digits.yhat1))
n10.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat2.train))
n10.outsample <- caret::confusionMatrix(xtabs(~digits.test.y + digits.yhat2))
n40.insample <- caret::confusionMatrix(xtabs(~digits.y + digits.yhat3.train))
n40.outsample <- caret::confusionMatrix(xtabs(~digits.test.y + digits.yhat3))
n40b.insample <- caret::confusionMatrix(xtabs(~digits.y + I(digits.yhat4.train - 1)))
n40b.outsample <- caret::confusionMatrix(xtabs(~ digits.test.y + I(digits.yhat4 - 1)))
shrinkage <- rbind(cbind(Size = 5, Sample = "In",
                         as.data.frame(t(n5.insample$overall[measures]))),
                   cbind(Size = 5, Sample = "Out",
                         as.data.frame(t(n5.outsample$overall[measures]))),
                   cbind(Size = 10, Sample = "In",
                         as.data.frame(t(n10.insample$overall[measures]))),
                   cbind(Size = 10, Sample = "Out",
                         as.data.frame(t(n10.outsample$overall[measures]))),
                   cbind(Size = 40, Sample = "In",
                         as.data.frame(t(n40.insample$overall[measures]))),
                   cbind(Size = 40, Sample = "Out",
                         as.data.frame(t(n40.outsample$overall[measures]))),
                   cbind(Size = 40, Sample = "In",
                         as.data.frame(t(n40b.insample$overall[measures]))),
                   cbind(Size = 40, Sample = "Out",
                         as.data.frame(t(n40b.outsample$overall[measures])))
)
shrinkage$Pkg <- rep(c("nnet", "RSNNS"), c(6, 2))

# 그림으로 나타내기
dodge <- position_dodge(width=0.4)
ggplot(shrinkage, aes(interaction(Size, Pkg, sep = " : "), Accuracy,
                      ymin = AccuracyLower, ymax = AccuracyUpper,
                      shape = Sample, linetype = Sample)) +
  geom_point(size = 2.5, position = dodge) +
  geom_errorbar(width = .25, position = dodge) +
  xlab("") + ylab("Accuracy + 95% CI") +
  theme_classic() +
  theme(legend.key.size = unit(1, "cm"), legend.position = c(.8, .2))










## 연습문제
# iris 데이터 학습 
set.seed(1234)
ind.iris <- sample(2, nrow(iris), replace = TRUE, prob = c(0.7, 0.3))
train.iris <- iris[ind.iris ==1,]
test.iris <- iris[ind.iris == 2,]
iris.nnet <- nnet(Species ~ ., train.iris, size = 20)

# 학습 데이터 정확도
(trTable.iris.nn <- table(predict(iris.nnet, type = "class"), train.iris$Species))
(trAccuracy.iris.nn <- sum(diag(trTable.iris.nn)) / sum(trTable.iris.nn))

# 테스트 데이터 정확도 
testPred.iris.nn <- predict(iris.nnet, newdata = test.iris, type = "class")
(teTable.iris.nn <- table(testPred.iris.nn, test.iris$Species))
(teAccuracy.iris.nn <- sum(diag(teTable.iris.nn)) / sum(teTable.iris.nn))
