library(keras)

# 데이터 준비
fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

image(train_images[1,,])
head(train_labels)

class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat', 
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')

dim(train_images)
dim(train_labels)
train_labels[1:20]

dim(test_images)
dim(test_labels)

## 데이터 전처리
# 이미지 보기
library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(train_images[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# 픽셀 값 수정
train_images <- train_images / 255
test_images <- test_images / 255

# 25개 이미지 확인
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}
par(mfcol=c(1,1))

## 신경망 모델 생성
# 레이어 구성
model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

# 모델 컴파일
model %>% compile(
  optimizer = 'adam', 
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

## 모델 학습
model %>% fit(train_images, train_labels, epochs = 5, verbose = 1)

model %>% fit(train_images, train_labels, validation_split = .3, epochs = 5, verbose = 1)

## 모델 평가
score <- model %>% evaluate(test_images, test_labels, verbose = 0)

cat('Test loss:', score["loss"], "\n")

cat('Test accuracy:', score["accuracy"], "\n")

## 예측
# 테스트 이미지들에 대한 예측
predictions <- model %>% predict(test_images)

predictions[1, ]

# 레이블은 0부터 시작, R 인덱스는 1부터 시작해서 1이 큼
which.max(predictions[1, ])

class_pred <- model %>% predict(test_images) %>% k_argmax()
class_pred[1:20]

class_pred2 <- RSNNS::encodeClassLabels(predictions) - 1
class_pred2[1:20]

test_labels[1]

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  #predicted_label <- which.max(predictions[i, ]) - 1
  predicted_label <- RSNNS::encodeClassLabels(matrix(predictions[i, ], nrow = 1), "WTA", h = 0.8, l = 0.2) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}
par(mfcol=c(1,1))

# 단일 이미지에 대한 예측
# Grab an image from the test dataset
# take care to keep the batch dimension, as this is expected by the model
img <- test_images[1, , ]
img <- test_images[1, , , drop = FALSE]
dim(img)

predictions <- model %>% predict(img)
predictions

# subtract 1 as labels are 0-based
which.max(predictions) - 1

class_pred <- model %>% predict(img) %>% k_argmax()
class_pred

k_argmax(predictions)

RSNNS::encodeClassLabels(matrix(predictions, nrow = 1)) - 1

test_labels[1]
table(test_labels)
