## 딥러닝 데이터 구조
library(mxnet)
ctx <- mx.cpu()
a <- mx.nd.ones(c(2, 3), ctx = ctx)
a
b <- a * 2 + 1
b
typeof(b)
class(b)

# 딥러닝 모델 생성
## UCI HAR 데이터
dataDirectory <- "../data"
link <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
download.file(link, destfile = paste(dataDirectory, "/HAR_Dataset.zip", sep=""))
unzip(paste(dataDirectory, '/HAR_Dataset.zip', sep=""), exdir = dataDirectory)

train.x <- read.table("../data/UCI HAR Dataset/train/X_train.txt")
train.y <- read.table("../data/UCI HAR Dataset/train/y_train.txt")[[1]]
test.x <- read.table("../data/UCI HAR Dataset/test/X_test.txt")
test.y <- read.table("../data/UCI HAR Dataset/test/y_test.txt")[[1]]
features <- read.table("../data/UCI HAR Dataset/features.txt")

## 데이터 변환 
meanSD <- grep("mean\\(\\)|std\\(\\)", features[, 2])
train.y <- train.y-1
test.y <- test.y-1

train.x <- t(train.x[,meanSD])
test.x <- t(test.x[,meanSD])
train.x <- data.matrix(train.x)
test.x <- data.matrix(test.x)

## 계산 그래프 정의
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=64)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=32)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=6)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

## 모델 실행
devices <- mx.cpu()
mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(softmax, X = train.x, y = train.y,
                                     ctx = devices,num.round = 20,
                                     learning.rate = 0.08, momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     initializer = mx.init.uniform(0.01),
                                     epoch.end.callback =
                                       mx.callback.log.train.metric(1))
print(proc.time() - tic)

## 모델 평가
preds1 <- predict(model, test.x)
pred.label <- max.col(t(preds1)) - 1
t <- table(data.frame(cbind(test.y,pred.label)),
           dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/length(test.y),2)
print(t)
print(sprintf(" Deep Learning Model accuracy = %1.2f%%",acc))

## Dunnhumby 데이터
library(readr)
library(reshape2)
library(dplyr)

source("ChapterIII/import.R")

# step 1, merge files
fileName <- import_data(data_directory,bExploreData=0)

# step 2, group and pivot data
fileOut <- paste(data_directory,"predict.csv",sep="")
df <- read_csv(fileName,col_types = cols(.default = col_character()))

#remove temp file
if (file.exists(fileName)) file.remove(fileName)

## 마감 날짜 설정
# convert spend to numeric field
df$SPEND<-as.numeric(df$SPEND)

# group sales by date. we have not converted the SHOP_DATE to date
# but since it is in yyyymmdd format,
# then ordering alphabetically will preserve date order
sumSalesByDate<-df %>%
  group_by(SHOP_WEEK,SHOP_DATE) %>%
  summarise(sales = sum(SPEND)
  )

# we want to get the cut-off date to create our data model
# this is the last date and go back 13 days beforehand
# therefore our X data only looks at everything from start to max date - 13 days
#  and our Y data only looks at everything from max date - 13 days to end (i.e. 14 days)
max(sumSalesByDate$SHOP_DATE)
sumSalesByDate2 <- sumSalesByDate[order(sumSalesByDate$SHOP_DATE),]
datCutOff <- as.character(sumSalesByDate2[(nrow(sumSalesByDate2)-13),]$SHOP_DATE)
datCutOff
rm(sumSalesByDate,sumSalesByDate2)

## 데이터 변환
# we are going to limit our data here from year 2008 only
# group data and then pivot it
sumTemp <- df %>%
  filter((SHOP_DATE < datCutOff) & (SHOP_WEEK>="200801")) %>%
  group_by(CUST_CODE,SHOP_WEEK,PROD_CODE_40) %>%
  summarise(sales = sum(SPEND)
  )
sumTemp$fieldName <- paste(sumTemp$PROD_CODE_40,sumTemp$SHOP_WEEK,sep="_")
df_X <- dcast(sumTemp,CUST_CODE ~ fieldName, value.var="sales")
df_X[is.na(df_X)] <- 0

## 목표 변수 생성
# y data just needs a group to get sales after cut-off date
df_Y <- df %>%
  filter(SHOP_DATE >= datCutOff) %>%
  group_by(CUST_CODE) %>%
  summarise(sales = sum(SPEND)
  )
colnames(df_Y)[2] <- "Y_numeric"

# use left join on X and Y data, need to include all values from X
#  even if there is no Y value
dfModelData <- merge(df_X,df_Y,by="CUST_CODE", all.x=TRUE)
# set binary flag
dfModelData$Y_categ <- 0
dfModelData[!is.na(dfModelData$Y_numeric),]$Y_categ <- 1
dfModelData[is.na(dfModelData$Y_numeric),]$Y_numeric <- 0
rm(df,df_X,df_Y,sumTemp)

nrow(dfModelData)
table(dfModelData$Y_categ)

# shuffle data
dfModelData <- dfModelData[sample(nrow(dfModelData)),]

write_csv(dfModelData,fileOut)

## 분류 데이터 로딩
library(randomForest)
library(xgboost)
library(ggplot2)

set.seed(42)
fileName <- "../data/dunnhumby/predict.csv"
dfData <- read_csv(fileName,
                   col_types = cols(
                     .default = col_double(),
                     CUST_CODE = col_character(),
                     Y_categ = col_integer())
)

## 데이터 분리
nobs <- nrow(dfData)
train <- sample(nobs, 0.9*nobs)
test <- setdiff(seq_len(nobs), train)
predictorCols <- colnames(dfData)[!(colnames(dfData) %in% c("CUST_CODE","Y_numeric","Y_categ"))]

trainData <- dfData[train, c(predictorCols)]
testData <- dfData[test, c(predictorCols)]
trainData$Y_categ <- dfData[train, "Y_categ"]$Y_categ
testData$Y_categ <- dfData[test, "Y_categ"]$Y_categ

## 벤치마크 데이터 생성
# 로지스틱 회귀 분석
logReg=glm(Y_categ ~ .,data=trainData,family=binomial(link="logit"))
pr <- as.vector(ifelse(predict(logReg, type="response",
                               testData) > 0.5, "1", "0"))
# Generate the confusion matrix showing counts.
t<-table(dfData[test, c(predictorCols, "Y_categ")]$"Y_categ", pr,
         dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/length(test),2)
print(t)
print(sprintf(" Logistic regression accuracy = %1.2f%%",acc))
rm(t,pr,acc)

# 랜덤 포레스트
rf <- randomForest::randomForest(as.factor(Y_categ) ~ .,
                                 data=trainData,
                                 na.action=randomForest::na.roughfix)
pr <- predict(rf, newdata=testData, type="class")
# Generate the confusion matrix showing counts.
t<-table(dfData[test, c(predictorCols, "Y_categ")]$Y_categ, pr,
         dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/length(test),2)
print(t)
print(sprintf(" Random Forest accuracy = %1.2f%%",acc))
rm(t,pr,acc)

# XGBoost
xgb <- xgboost(data=data.matrix(trainData[,predictorCols]), label=trainData[,"Y_categ"]$Y_categ,
               nrounds=75, objective="binary:logistic")
pr <- as.vector(ifelse(
  predict(xgb, data.matrix(testData[, predictorCols])) > 0.5, "1", "0"))
t<-table(dfData[test, c(predictorCols, "Y_categ")]$"Y_categ", pr,
         dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/length(test),2)
print(t)
print(sprintf(" XGBoost accuracy = %1.2f%%",acc))
rm(t,pr,acc)

############
##딥러닝 모델 생성
############
require(mxnet)

# MXNet expects matrices
train_X <- data.matrix(trainData[, predictorCols])
test_X <- data.matrix(testData[, predictorCols])
train_Y <- trainData$Y_categ

# hyper-parameters
num_hidden <- c(128,64,32)
drop_out <- c(0.2,0.2,0.2)
wd=0.00001
lr <- 0.03
num_epochs <- 40
activ <- "relu"

# create our model architecture
# using the hyper-parameters defined above
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=num_hidden[1])
act1 <- mx.symbol.Activation(fc1, name="activ1", act_type=activ)

drop1 <- mx.symbol.Dropout(data=act1,p=drop_out[1])
fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=num_hidden[2])
act2 <- mx.symbol.Activation(fc2, name="activ2", act_type=activ)

drop2 <- mx.symbol.Dropout(data=act2,p=drop_out[2])
fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=num_hidden[3])
act3 <- mx.symbol.Activation(fc3, name="activ3", act_type=activ)

drop3 <- mx.symbol.Dropout(data=act3,p=drop_out[3])
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc4, name="sm")

# run on cpu, change to 'devices <- mx.gpu()'
#  if you have a suitable GPU card
devices <- mx.cpu()
mx.set.seed(0)
tic <- proc.time()
# This actually trains the model
model <- mx.model.FeedForward.create(softmax, X = train_X, y = train_Y,
                                     ctx = devices,num.round = num_epochs,
                                     learning.rate = lr, momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     initializer = mx.init.uniform(0.1),
                                     wd=wd,
                                     epoch.end.callback = mx.callback.log.train.metric(1))
print(proc.time() - tic)

pr <- predict(model, test_X)
pred.label <- max.col(t(pr)) - 1
t <- table(data.frame(cbind(testData[,"Y_categ"]$Y_categ,pred.label)),
           dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/length(test),2)
print(t)
print(sprintf(" Deep Learning Model accuracy = %1.2f%%",acc))
rm(t,pr,acc)
rm(data,fc1,act1,fc2,act2,fc3,act3,fc4,softmax,model)




















######################################
## 연습문제
######################################
# hyper-parameters
num_hidden <- c(256,128,64,32)
drop_out <- c(0.2,0.2,0.1,0.1)
wd=0.0
lr <- 0.03
num_epochs <- 50
activ <- "relu"

# create our model architecture
# using the hyper-parameters defined above
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=num_hidden[1])
act1 <- mx.symbol.Activation(fc1, name="activ1", act_type=activ)

drop1 <- mx.symbol.Dropout(data=act1,p=drop_out[1])
fc2 <- mx.symbol.FullyConnected(drop1, name="fc2", num_hidden=num_hidden[2])
act2 <- mx.symbol.Activation(fc2, name="activ2", act_type=activ)

drop2 <- mx.symbol.Dropout(data=act2,p=drop_out[2])
fc3 <- mx.symbol.FullyConnected(drop2, name="fc3", num_hidden=num_hidden[3])
act3 <- mx.symbol.Activation(fc3, name="activ3", act_type=activ)

drop3 <- mx.symbol.Dropout(data=act3,p=drop_out[3])
fc4 <- mx.symbol.FullyConnected(drop3, name="fc4", num_hidden=num_hidden[4])
act4 <- mx.symbol.Activation(fc4, name="activ4", act_type=activ)
drop4 <- mx.symbol.Dropout(data=act4,p=drop_out[4])

fc5 <- mx.symbol.FullyConnected(drop4, name="fc5", num_hidden=2)
softmax <- mx.symbol.SoftmaxOutput(fc5, name="sm")

# run on cpu, change to 'devices <- mx.gpu()'
#  if you have a suitable GPU card
devices <- mx.cpu()
mx.set.seed(0)
tic <- proc.time()
# This actually trains the model
model <- mx.model.FeedForward.create(softmax, X = train_X, y = train_Y,
                                     ctx = devices,num.round = num_epochs,
                                     learning.rate = lr, momentum = 0.9,
                                     eval.metric = mx.metric.accuracy,
                                     initializer = mx.init.uniform(0.1),
                                     wd=wd,
                                     epoch.end.callback = mx.callback.log.train.metric(1))
print(proc.time() - tic)

pr <- predict(model, test_X)
pred.label <- max.col(t(pr)) - 1
t <- table(data.frame(cbind(testData[,"Y_categ"]$Y_categ,pred.label)),
           dnn=c("Actual", "Predicted"))
acc<-round(100.0*sum(diag(t))/length(test),2)
print(t)
print(sprintf(" Deep Learning Model accuracy = %1.2f%%",acc))
rm(t,pr,acc)
rm(data,fc1,act1,fc2,act2,fc3,act3,fc4,act4,fc5,softmax,model)
