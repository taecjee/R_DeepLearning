library(keras)

library(tensorflow)

## GPU 사용 - 메모리 관리
gpus <- tf$config$experimental$list_physical_devices("GPU")
tf$config$experimental$set_memory_growth(gpus[[1]], TRUE)

# MNIST 데이터 로딩
mnist <- dataset_mnist()
image(mnist$train$x[1,,])

mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255

# Keras 모델 정의
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

summary(model)

# 모델 컴파일
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

# 모델 학습
model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )

# 모델 성능 평가
predictions <- predict(model, mnist$test$x)
head(predictions, 2)

library(RSNNS)
mnist.test.yhat <- encodeClassLabels(predictions, method = "WTA", l = 0, h = 0.8) - 1
table(mnist$test$y, mnist.test.yhat)
mnist.test.yhat2 <- encodeClassLabels(predictions) - 1
caret::confusionMatrix(xtabs(~as.matrix(mnist$test$y) + as.matrix(mnist.test.yhat2)))

model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

# 모델 저장 및 재사용
save_model_tf(object = model, filepath = "model/ver2")

reloaded_model2 <- load_model_tf("model/ver2")
all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))
