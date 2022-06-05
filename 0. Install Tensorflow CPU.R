install.packages("remotes")
install.packages("reticualte")
library(reticulate)
conda_create("tensorflow_1")

reticulate::use_condaenv(condaenv = "tensorflow_1",required = TRUE)
py_config()

conda_install("tensorflow_1", "scipy")
conda_install("tensorflow_1", packages = c("pandas","scikit-learn", "tensorflow"))

install.packages("tensorflow")

library(tensorflow)
tf$constant("Hellow Tensorflow")

install.packages("keras")
library(keras)
