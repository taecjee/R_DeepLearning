## 컨볼루션 계산
a <- matrix(c(0.5, 0.3, 0.1, 0.2, 0.6, 0.1, 0.1, 0.1, 0.7), nrow = 3, byrow = TRUE)
b <- matrix(c(0.5, 0.6, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0), nrow = 3, byrow = TRUE)

conv <- matrix(c(3, 1, 1, 1, 3, 1, 1, 1, 3), nrow = 3)

a * conv
sum(a*conv)

b * conv
sum(b * conv)


## CIFAR10 이미지 분류하기
library(tensorflow)
library(keras)

# CIFAR10 데이터 준비
cifar <- dataset_cifar10()

class_names <- c('airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')

index <- 1:30

par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))
cifar$train$x[index,,,] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(class_names[cifar$train$y[index] + 1]) %>% 
  purrr::map(as.raster, max = 255) %>%
  purrr::iwalk(~{plot(.x); title(.y)})
par(mfcol = c(1,1))

# 모델 생성
# Convolution 레이어
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(32,32,3)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu")

summary(model)

# Dense 레이어
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)

# 모델 학습
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% 
  fit(
    x = cifar$train$x, y = cifar$train$y,
    epochs = 10,
    validation_data = unname(cifar$test),
    verbose = 2
  )

# 모델 평가
plot(history)

evaluate(model, cifar$test$x, cifar$test$y, verbose = 0)

evaluate(model, cifar$train$x, cifar$train$y, verbose = 0)



























##########################################
## 연습문제
##########################################
library(keras)

# 데이터 준비
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

dim(train_images)
dim(test_images)

train_images2 <- array_reshape(train_images, dim = c(60000, 28, 28, 1))
dim(train_images2)
test_images2 <- array_reshape(test_images, dim = c(10000, 28, 28, 1))
dim(test_images2)

# Convolution Layer
model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu", 
                input_shape = c(28,28,1)) %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2,2)) %>% 
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "relu")

summary(model)

# Dense Layer
model %>% 
  layer_flatten() %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 10, activation = "softmax")

summary(model)

# 모델 학습
model %>% compile(
  optimizer = "adam",
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% 
  fit(
    x = train_images2, y = train_labels,
    epochs = 10,
    validation_split = 0.3,
    verbose = 2
  )

# 모델 평가
preds2 <- model %>% predict(test_images2) %>% k_argmax()

preds2.array <- as.array(preds2)

res2 <- data.frame(cbind(test_labels, preds2.array))

table(res2)
accuracy2 <- sum(res2$test_labels == res2$preds2.array) / nrow(res2)
accuracy2