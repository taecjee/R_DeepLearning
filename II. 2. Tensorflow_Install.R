library(reticulate)
conda_create("tensorflow_1")
use_condaenv(condaenv = "tensorflow_1", required = TRUE)
py_config()

conda_install("tensorflow_1", "scipy")
conda_install("tensorflow_1", packages = c("numpy", "pandas", "scikit-learn", "tensorflow"))
conda_install("tensorflow_1", "keras")

library(tensorflow)
tf$constant("Hello")
