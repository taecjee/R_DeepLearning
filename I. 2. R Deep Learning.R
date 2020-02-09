#######################
#### 연습 문제
#######################
# MXNet
# Install the MXNet package for Windows
# https://mxnet.incubator.apache.org/install/index.html
# Note: packages for 3.6.x are not yet available. Install 3.5.x of R from CRAN.
cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN"
options(repos = cran)
install.packages("mxnet")

# Install Keras for windows
# https://kera.rstudio.com
# 1. install Rtools (https://cran.r-project.org/bin/windows/Rtools/)
# 2. Install Anaconda for Python 3.x (https://www.anaconda.com/download/#windows)
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/keras")
library(keras)
install_keras()

