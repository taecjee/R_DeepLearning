library(reticulate)
conda_create("tensorflow2_2")
use_condaenv(condaenv = "tensorflow2_2", required = TRUE)
py_config()

conda_install("tensorflow2_2", "scipy")
conda_install("tensorflow2_2", packages = c("numpy", "pandas", "scikit-learn", "tensorflow"))
conda_install("tensorflow2_2", "keras")

library(tensorflow)
tf$constant("Hello")
