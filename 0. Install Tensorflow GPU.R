# CUDA, CUDNN 설치
# 환경 설정 추가
# https://www.tensorflow.org/install/gpu#hardware_requirements

library(reticulate)
conda_create("tensorflow_2")

reticulate::use_condaenv(condaenv = "tensorflow_2",required = TRUE)
py_config()

conda_install("tensorflow_2", "scipy")
conda_install("tensorflow_2", packages = c("pandas","scikit-learn"))
reticulate::conda_install("tensorflow_2", "cudatoolkit")
reticulate::conda_install("tensorflow_2", "cudnn")
reticulate::conda_install("tensorflow_2", "tensorflow-gpu")

library(keras)
library(tensorflow)

## GPU 사용 - 메모리 관리
gpus <- tf$config$experimental$list_physical_devices("GPU")
tf$config$experimental$set_memory_growth(gpus[[1]], TRUE)

tf$constant("Hellow Tensorflow")
