## tensorflow 기본
library(tensorflow)

# tensor of rank-0
var1 <- tf$constant(0.1)
var2 <- tf$constant(2.3)
var3 <- var1 + var2
print(var1)

library(keras)
var1 %>% k_eval()
var2 %>% k_eval()
var3 %>% k_eval()

# tensor of rank-1
var4 <- tf$constant(4.5,shape=shape(5L))
var4 %>% k_eval()

# tensor of rank-2
var5 <- tf$constant(6.7,shape=shape(3L,3L))
var5 %>% k_eval()

## 선형회귀
# 가상 데이터 생성
set.seed(42)
# create 50000 x variable between 0 and 100
x_var <- runif(50000,min=0,max=1)
#y = approx(1.3x + 0.8)
y_var <- rnorm(50000,0.8,0.04) + x_var * rnorm(50000,1.3,0.05)
# y_pred = beta0 + beta1 * x
beta0 <- tf$Variable(tf$zeros(shape(1L)))
beta1 <- tf$Variable(tf$random$uniform(shape(1L), -1.0, 1.0))
y_pred <- beta0 + beta1*x_var

# loss 함수 설정
# create our loss value which we want to minimize
loss <- tf$reduce_mean((y_pred-y_var)^2)
# create optimizer
optimizer <- tf$compat$v1$train$GradientDescentOptimizer(0.6)
train <- optimizer$minimize(loss)

# 모델 실행
# create TensorFlow session and initialize variables
sess = tf$Session()
sess$run(tf$global_variables_initializer())
# solve the regression
for (step in 0:80) {
  if (step %% 10 == 0)
    print(sprintf("Step %1.0f:beta0=%1.4f,
beta1=%1.4f",step,sess$run(beta0), sess$run(beta1)))
  sess$run(train)
}

## CNN
# MNIST 데이터
library(RSNNS)

mnist <- dataset_mnist()
set.seed(42)
xtrain <- array_reshape(mnist$train$x,c(nrow(mnist$train$x),28*28))
ytrain <- decodeClassLabels(mnist$train$y)
xtest <- array_reshape(mnist$test$x,c(nrow(mnist$test$x),28*28))
ytest <- decodeClassLabels(mnist$test$y)
xtrain <- xtrain / 255.0
xtest <- xtest / 255.0
head(ytrain)
head(mnist$train$y)

# placeholders
x <- tf$placeholder(tf$float32, shape(NULL,28L*28L))
y <- tf$placeholder(tf$float32, shape(NULL,10L))
x_image <- tf$reshape(x, shape(-1L,28L,28L,1L))

# 모델 정의
# first convolution layer
conv_weights1 <- tf$Variable(tf$random_uniform(shape(5L,5L,1L,16L), -0.4, 0.4))
conv_bias1 <- tf$constant(0.0, shape=shape(16L))
conv_activ1 <- tf$nn$tanh(tf$nn$conv2d(x_image, conv_weights1,
                                       strides=c(1L,1L,1L,1L), padding='SAME') + conv_bias1)
pool1 <- tf$nn$max_pool2d(conv_activ1,
                        ksize=c(1L,2L,2L,1L),strides=c(1L,2L,2L,1L), padding='SAME')
# second convolution layer
conv_weights2 <- tf$Variable(tf$random_uniform(shape(5L,5L,16L,32L), -0.4, 0.4))
conv_bias2 <- tf$constant(0.0, shape=shape(32L))
conv_activ2 <- tf$nn$relu(tf$nn$conv2d(pool1, conv_weights2,
                                       strides=c(1L,1L,1L,1L), padding='SAME') + conv_bias2)
pool2 <- tf$nn$max_pool2d(conv_activ2,
                        ksize=c(1L,2L,2L,1L),strides=c(1L,2L,2L,1L), padding='SAME')
# densely connected layer
dense_weights1 <- tf$Variable(tf$truncated_normal(shape(7L*7L*32L,512L), stddev=0.1))
dense_bias1 <- tf$constant(0.0, shape=shape(512L))
pool2_flat <- tf$reshape(pool2, shape(-1L,7L*7L*32L))
dense1 <- tf$nn$relu(tf$matmul(pool2_flat, dense_weights1) + dense_bias1)
# dropout
keep_prob <- tf$placeholder(tf$float32)
dense1_drop <- tf$nn$dropout(dense1, keep_prob)
# softmax layer
dense_weights2 <- tf$Variable(tf$truncated_normal(shape(512L,10L), stddev=0.1))
dense_bias2 <- tf$constant(0.0, shape=shape(10L))
yconv <- tf$nn$softmax(tf$matmul(dense1_drop, dense_weights2) + dense_bias2)

## 하이퍼 파라미터 정의
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y * tf$log(yconv), reduction_indices=1L))
train_step <- tf$compat$v1$train$AdamOptimizer(0.0001)$minimize(cross_entropy)
correct_prediction <- tf$equal(tf$argmax(yconv, 1L), tf$argmax(y, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

## 모델 학습
sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())
# if you get out of memory errors when running on gpu
# then lower the batch_size
batch_size <- 128
batches_per_epoch <- 1+nrow(xtrain) %/% batch_size
for (epoch in 1:10)
{
  for (batch_no in 0:(-1+batches_per_epoch))
  {
    nStartIndex <- 1 + batch_no*batch_size
    nEndIndex <- nStartIndex + batch_size-1
    if (nEndIndex > nrow(xtrain))
      nEndIndex <- nrow(xtrain)
    xvalues <- xtrain[nStartIndex:nEndIndex,]
    yvalues <- ytrain[nStartIndex:nEndIndex,]
    if (batch_no %% 100 == 0) {
      batch_acc <-
        accuracy$eval(feed_dict=dict(x=xvalues,y=yvalues,keep_prob=1.0))
      print(sprintf("Epoch %1.0f, step %1.0f: training accuracy=%1.4f",epoch, batch_no, batch_acc))
    }
    sess$run(train_step,feed_dict=dict(x=xvalues,y=yvalues,keep_prob=0.5))
  }
  cat("\n")
}

# 모델 평가
# calculate test accuracy
# have to run in batches to prevent out of memory errors
batches_per_epoch <- 1+nrow(xtest) %/% batch_size
test_acc <- vector(mode="numeric", length=batches_per_epoch)
for (batch_no in 0:(-1+batches_per_epoch))
{
  nStartIndex <- 1 + batch_no*batch_size
  nEndIndex <- nStartIndex + batch_size-1
  if (nEndIndex > nrow(xtest))
    nEndIndex <- nrow(xtest)
  xvalues <- xtest[nStartIndex:nEndIndex,]
  yvalues <- ytest[nStartIndex:nEndIndex,]
  batch_acc <-
    accuracy$eval(feed_dict=dict(x=xvalues,y=yvalues,keep_prob=1.0))
  test_acc[batch_no+1] <- batch_acc
}
# using the mean is not totally accurate as last batch is not a complete batch
print(sprintf("Test accuracy=%1.4f",mean(test_acc)))

## keras
library(keras)

mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters=32, kernel_size=c(3,3), activation='relu',
                input_shape=c(28, 28, 1)) %>% 
  layer_conv_2d(filters=64, kernel_size=c(3,3), activation='relu') %>% 
  layer_max_pooling_2d(pool_size=c(2, 2)) %>% 
  layer_dropout(rate=0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units=128, activation='relu') %>% 
  layer_dropout(rate=0.5) %>% 
  layer_dense(units=10, activation='softmax')

model %>% compile(
  loss=loss_categorical_crossentropy,
  optimizer="rmsprop",
  metrics=c('accuracy')
)

batch_size <- 128
epochs <- 5
model %>% fit(
  x_train, y_train,
  batch_size=batch_size,
  epochs=epochs,
  verbose=1,
  callbacks = callback_tensorboard("/tensorflow_logs",
                                   histogram_freq=1,write_images=0),
  validation_split = 0.2
)
# from cmd line,run 'tensorboard --logdir /tensorflow_logs'

scores <- model %>% evaluate(x_test, y_test,verbose=0)
print(sprintf("test accuracy = %1.4f",scores[[2]]))
