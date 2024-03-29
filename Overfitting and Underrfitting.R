library(keras)
library(dplyr)
library(ggplot2)
library(tidyr)
library(tibble)

# Download the IMDB dataset
num_words <- 1000
imdb <- dataset_imdb(num_words = num_words)

c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test

multi_hot_sequences <- function(sequences, dimension) {
  multi_hot <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) {
    multi_hot[i, sequences[[i]]] <- 1
  }
  multi_hot
}

train_data <- multi_hot_sequences(train_data, num_words)
test_data <- multi_hot_sequences(test_data, num_words)

first_text <- data.frame(word = 1:num_words, value = train_data[1, ])
ggplot(first_text, aes(x = word, y = value)) +
  geom_line() +
  theme(axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())

## Demonstrate overfitting
# CREATE A BASELINE MODEL
baseline_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

baseline_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(baseline_model)

baseline_history <- baseline_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 1
)

# CREATE A SMALLER MODEL
smaller_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 4, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

smaller_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(smaller_model)

smaller_history <- smaller_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

# CREATE A BIGGER MODEL
bigger_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

bigger_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

summary(bigger_model)

bigger_history <- bigger_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

# PLOT THE TRAINING AND VALIDATION LOSS
compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  bigger_train = bigger_history$metrics$loss,
  bigger_val = bigger_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")

## Strategies
# ADD WEIGHT REGULARIZATION
l2_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words,
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 1, activation = "sigmoid")

l2_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

l2_history <- l2_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  l2_train = l2_history$metrics$loss,
  l2_val = l2_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")

# ADD DROPOUT
dropout_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 1, activation = "sigmoid")

dropout_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

dropout_history <- dropout_model %>% fit(
  train_data,
  train_labels,
  epochs = 40,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  l2_train = l2_history$metrics$loss,
  l2_val = l2_history$metrics$val_loss,
  dropout_train = dropout_history$metrics$loss,
  dropout_val = dropout_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")



##############################
# ADD WEIGHT REGULARIZATION & Drop out
l2_do_model <- 
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words,
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(0.4) %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(0.4) %>%
  layer_dense(units = 1, activation = "sigmoid")

l2_do_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

l2_do_history <- l2_do_model %>% fit(
  train_data,
  train_labels,
  epochs = 20,
  batch_size = 512,
  validation_data = list(test_data, test_labels),
  verbose = 2
)

compare_cx <- data.frame(
  baseline_train = baseline_history$metrics$loss,
  baseline_val = baseline_history$metrics$val_loss,
  smaller_train = smaller_history$metrics$loss,
  smaller_val = smaller_history$metrics$val_loss,
  l2_do_train = l2_history$metrics$loss,
  l2_do_val = l2_history$metrics$val_loss
) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)

ggplot(compare_cx, aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")
