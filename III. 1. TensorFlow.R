library(keras)

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

model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

# 모델 저장 및 재사용
save_model_tf(object = model, filepath = "model")

reloaded_model <- load_model_tf("model")
all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))
