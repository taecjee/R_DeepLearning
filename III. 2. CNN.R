## 컨볼루션 계산
a <- matrix(c(0.5, 0.3, 0.1, 0.2, 0.6, 0.1, 0.1, 0.1, 0.7), nrow = 3, byrow = TRUE)
b <- matrix(c(0.5, 0.6, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0), nrow = 3, byrow = TRUE)

conv <- matrix(c(3, 1, 1, 1, 3, 1, 1, 1, 3), nrow = 3)

a * conv
sum(a*conv)

b * conv
sum(b * conv)
