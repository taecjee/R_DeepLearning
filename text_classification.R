library(keras)
library(dplyr)
library(ggplot2)
library(purrr)

#install.packages("pins")
#library(pins)
# Download the Movie Reviews dataset
#paths <- pins::pin_get("nltkdata/movie-review", "kaggle")
# we only need the movie_review.csv file
#path <- paths[1]
path <- "E:/R_Project/R_DeepLearning/data/movie_review.csv"

df <- readr::read_csv(path)

head(df)

# Explore the data
df %>% count(tag)

df$text[1]

training_id <- sample.int(nrow(df), size = nrow(df)*0.8)
training <- df[training_id,]
testing <- df[-training_id,]

df$text %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()

# Prepare the data
num_words <- 10000
max_length <- 50
text_vectorization <- layer_text_vectorization(
  max_tokens = num_words, 
  output_sequence_length = max_length, 
)

text_vectorization %>% 
  adapt(df$text)

get_vocabulary(text_vectorization)

text_vectorization(matrix(df$text[1], ncol = 1))

# Build the model
input <- layer_input(shape = c(1), dtype = "string")

output <- input %>% 
  text_vectorization() %>% 
  layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
  layer_global_average_pooling_1d() %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model <- keras_model(input, output)

summary(model)

model %>% compile(
  optimizer = 'adam',
  loss = 'binary_crossentropy',
  metrics = list('accuracy')
)

# Train the model
history <- model %>% fit(
  training$text,
  as.numeric(training$tag == "pos"),
  epochs = 20,
  batch_size = 512,
  validation_split = 0.2,
  verbose=1
)

# Evaluate the model
results <- model %>% evaluate(testing$text, as.numeric(testing$tag == "pos"), verbose = 0)
results
