# 동작 확인 및 명령 실행
print('hello')
print(
'hello'
)

# 도움말 보기
?print
help('print')

# 패키지 설치 - 온라인
install.packages('e1071')

# 패키지 설치 - 오프라인
packageDist <- ('package\\e1071_1.7-3.zip')
install.packages(packageDist, repos=NULL)

# 변수
var1 <- 10
var1 <- 'a string'

한글 <- 1
b <- 2
a1 <- 3
.x <- 4 # Values 창에는 보이지 않음
y_1 <- 5

2a <- 6
_a <- 9
.2a <- 10

# NA와 NULL
four <- NA
is.na(four)

one <- 100
is.null(one)
two <- as.null(one)
is.null(two)

# 문자열 (Character)
a <- 'This is a string'
print(a)

b <- "This is a string"
print(b)

# 소수 숫자 (Numeric)
a <- 4.5
b <- 3.6
c <- a + b
print(c)

class(c)
typeof(c)

# 정수 숫자 (Integer)
a <- 3
class(a)
b <- as.integer(a)
class(b)

b <- a + b
class(b)

# 진리값 (Logical)
TRUE & TRUE
TRUE & FALSE
TRUE | FALSE
FALSE | FALSE
!TRUE
!FALSE

# 요인 (factor)
animals <- factor("dog", c('cat', 'dog', 'horse'))
animals

nlevels(animals)
levels(animals)

factor(c('c', 'a', 'b'), ordered = TRUE)
ordered(c('low', 'high', 'middle', 'low', 'middle'))
ordered(c('low', 'high', 'middle', 'low', 'middle'), levels = c('low', 'middle', 'high'))

# 벡터 (Vector)
aaa <- numeric(length = 5)
aaa[1] <- 6
aaa[2] <- 2
class(aaa)
aaa[1] - aaa[2]
aaa[3] <- 'a string'
class(aaa)
aaa[1] - aaa[2]

x <- c(1, 2, 3, 4)
x
names(x) <- c('Sunday', 'Monday', 'Tuesday', 'Wednesday')
x

# 벡터 내 데이터 접근
x[1]
x[-1]
x[c(1, 3, 4)]
x[2:4]
x['Monday']
x[c('Tuesday', 'Sunday')]
length(x)
NROW(x)
nrow(x)
y <- c()
y
x[10]

# 벡터의 특수 형태
seq(1, 10)
seq(1, 10, 2)
1:10

rep(1, 5)
rep(1:2, 5)
rep(1:2, each = 5)
rep(c(1,5), 3)
rep(c(1,5), 3, each = 4)

# 리스트 (List)
aaa <- list()
aaa[1] <- 4
aaa[2] <- 5
aaa[3] <- 'a string'
aaa

bbb <- list(name = 'dog1', height = 60)
bbb

bbb$name
bbb$height
bbb[[1]]

bbb[1]

# 행렬 (Matrix)
numeric.vector <- 1:20
numeric.vector
numeric.mat <- matrix(numeric.vector, 4, 5)
class(numeric.mat)
numeric.mat

matrix(numeric.vector, nrow = 4)
matrix(numeric.vector, ncol = 5)
matrix(numeric.vector, ncol = 5, byrow = T)

numeric.mat <- matrix(numeric.vector, ncol = 5, dimnames=list(c('a', 'b', 'c', 'd'), c('A', 'B', 'C', 'D', 'E')))
numeric.mat
colnames(numeric.mat)
rownames(numeric.mat)

# 행렬 (Matrix) 내 데이터 접근
numeric.mat[1, 1]
numeric.mat[2, 3]

numeric.mat[1:2, ]
numeric.mat[-3, ]
numeric.mat[c(1,3), c(3,1)]

numeric.mat['a', c('A', 'C')]

# 데이터 프레임 (Data Frame)
numeric.vector <- 1:5
character.vector <- letters[1:5]
class(numeric.vector)
class(character.vector)
df <- data.frame(x = numeric.vector, y = character.vector)
df
class(df)

df$v <- c('M', 'F', 'M', 'F', 'F')
df

df$x
df[1,]
df[2,3]

# 데이터 프레임 (Data Frame) 관련 함수
str(df)
head(df, 2)
tail(df, 2)

colnames(df)
colnames(df) <- c('first', 'second', 'third')
colnames(df)

rownames(df) <- letters[1:5]
df

# 데이터 유형 확인 및 변환
class(df)
class(df$first)
class(df$third)

is.numeric(df$first)
is.numeric(df$third)
is.data.frame(df)

as.factor(df$third)
as.numeric(df$third)
as.numeric(as.factor(df$third))
as.matrix(df)

# 제어문
if (TRUE) {
  print ('TRUE')
} else {
  print ('FALSE')
}

if (TRUE) {
  print ('TRUE')
}
else {
  print('FALSE')
}

# 반복문
for (i in 1:5) {
  print (i)
}

for (i in df$second) {
  print (i)
}

i <- 0
while (i < 5) {
  print (i)
  i <- i + 1
}

# 결측치의 처리
NA & T
NA + 1

sum(c(1, 2, 3, 4, NA))
sum(c(1, 2, 3, 4, NA), na.rm = T)

x <- data.frame(a = c(1, 2, 3), b = c('a', 'b', NA), c = c(4, NA, 6))
x
na.omit(x)
na.pass(x)
na.fail(x)

# 2차원 데이터
df1 <- read.table("data/data2.txt")
df1
class(df1$V2)

df1 <- read.table("data/data2.txt", stringsAsFactors = FALSE)
df1
class(df1$V2)

# 텍스트 파일 저장
head(trees)

write.table(trees, "data/out1.txt")
write.table(trees, "data/out2.txt", quote = FALSE)
write.table(trees, "data/out3.txt", quote = FALSE, row.names = FALSE)
write.table(trees, "data/out4.txt", quote = FALSE, row.names = FALSE, sep = ",")
write.csv(trees, "data/out1.csv")
write.csv(trees, "data/out2.csv", quote = FALSE, row.names = FALSE)

# readr 패키지 활용
install.packages("readr")
library(readr)

read_csv("data/data2.txt")

# Excel 파일 불러오기
install.packages("xlsx")
install.packages("readxl")

library(xlsx)
library(readxl)

xls_file <- system.file("tests", "test_import.xlsx", package = "xlsx")
xls1 <- read.xlsx(xls_file, sheetIndex = 1)
head(xls1)

xls2 <- read.xlsx(xls_file, sheetIndex = 1, rowIndex = 1:5, colIndex = 1:2)
xls2

xls3 <- read_excel(xls_file)
head(xls3)

## dplyr
# 조건에 따른 관찰값의 선택

install.packages("dplyr")
library(dplyr)

filter(mtcars, mpg >= 30)

idx <- which(mtcars$mpg >= 30)
mtcars[idx,]

filter(mtcars, mpg >= 30 & wt < 1.8)

filter(mtcars, mpg <= 30 & (cyl == 6 | cyl == 8) & am == 1)
filter(mtcars, mpg <= 30 & cyl %in% c(6,8) & am == 1)
filter(mtcars, mpg <= 30, cyl %in% c(6,8), am == 1)

filter(mtcars, mpg >= median(mpg) & mpg <= quantile(mpg, probs = 0.75))
filter(mtcars, between(mpg, median(mpg), quantile(mpg, probs = 0.75)))

# 관찰값의 단순 임의 추출
sample_n(mtcars, size = 3)
nrow(mtcars)
sample_frac(mtcars, size = 0.1)

myIdx <- sample(1:nrow(mtcars), size = 3)
mtcars[myIdx,]

# 특정 변수의 값이 가장 큰(작은) 관찰값 선택
top_n(mtcars, n = 2, wt=mpg)
top_n(mtcars, n = -2, wt=mpg)

# 관찰값의 정렬
arrange(mtcars, mpg)
arrange(mtcars, desc(mpg))
arrange(mtcars, mpg, desc(wt))

# 변수의 선택
select(mtcars, mpg, cyl, disp)
select(mtcars, mpg:disp)
select(mtcars, 1:3)

select(mtcars, -mpg)
select(mtcars, -mpg, -cyl, -disp)
select(mtcars, -(mpg:disp))
select(mtcars, -(1:3))

select(mtcars, starts_with("d"))
select(mtcars, ends_with("t"))
select(mtcars, contains("a"))

# 변수의 선택, 변수 이름 수정
select(mtcars, contains("ar", ignore.case = FALSE))

select(mtcars, -starts_with("d"))
select(mtcars, -contains("a"))

select(mtcars, vs, wt, everything())

# 변수 이름 수정
rename(mtcars, MPG = mpg)

# 새로운 변수의 추가
mutate(mtcars, 
       kml = mpg * 0.43, 
       gp_kml = if_else(kml >= 10, "good", "bad"))

transmute(mtcars, 
          kml = mpg * 0.43, 
          gp_kml = if_else(kml >= 10, "good", "bad"))

# 그룹 생성 및 그룹 별 자료 요약
summarise(mtcars, avg_mpg = mean(mpg))
summarise(mtcars, n = n(), n_mpg = n_distinct(mpg), avg_mpg = mean(mpg), sd_mpg = sd(mpg))

by_cyl <- group_by(mtcars, cyl)
by_cyl
summarise(by_cyl, n = n(), n_mpg = n_distinct(mpg), avg_mpg = mean(mpg), sd_mpg = sd(mpg))

# Pipe 기능
mtcars %>% 
  group_by(cyl) %>% 
  summarise(n = n(), n_mpg = n_distinct(mpg), avg_mpg = mean(mpg), sd_mpg = sd(mpg))

mtcars %>% 
  mutate(kml = mpg * 0.43,
         gp_kml = if_else(kml >= 10, "good", "bad")) %>% 
  select(mpg, kml, gp_kml, everything()) %>% 
  filter(gear >= 4) %>% 
  group_by(cyl, gp_kml) %>% 
  summarise(n = n(),
            avg_mpg = mean(mpg),
            avg_kml = mean(kml))

## tidyr
# gather로 tidy 데이터 만들기

install.packages("tidyr")
library(tidyr)

table1

table4a

table4a %>% 
  gather(key = year, value = cases, `1999`, `2000`)

table4a %>% 
  gather(key = year, value = cases, `1999`, `2000`) %>% 
  arrange(country)

# spread로 tidy 데이터 만들기
table1

table2

table2 %>% 
  spread(key = type, value = count)

# separate로 tidy 데이터 만들기
table1

table3

table3 %>% 
  separate(col = rate, into = c("cases", "population"))

table3 %>% 
  separate(col = rate, into = c("cases", "population"),
           sep = 4)

table3 %>% 
  separate(col = rate, into = c("cases", "population"),
           convert = TRUE)

# unite로 데이터 다루기
table1

table5

table5 %>% 
  unite(col = year, century, year)

table5 %>% 
  unite(col = year, century, year, sep="") %>% 
  separate(col = rate, into = c("cases", "population"),
           convert = TRUE) %>% 
  mutate(year = as.integer(year))

## 데이터 결합
# Mutating joins
band_members
band_instruments

band_members %>% 
  inner_join(band_instruments)

band_members %>% 
  left_join(band_instruments, by = "name")

band_members %>% 
  right_join(band_instruments, by = "name")

band_members %>% 
  full_join(band_instruments, by = "name")

band_instruments2
band_members %>% 
  full_join(band_instruments2, by = c("name" = "artist"))

# Filtering joins
band_members
band_instruments

band_members %>% 
  inner_join(band_instruments)

band_members %>% 
  semi_join(band_instruments)

band_members %>% 
  anti_join(band_instruments, by = "name")

# 단순 수평 및 수직 결합
df_x <- tibble(x1 = letters[1:3],
               x2 = 1:3)
df_y <- tibble(y1 = LETTERS[4:6],
               y2 = 4:6)

bind_cols(df_x, df_y)

bind_rows(df_x, df_y)

df_z <- tibble(x1 = LETTERS[4:6],
               x2 = 4:6)

bind_rows(df_x, df_z)


#######################
#### 연습 문제
#######################
# 다음과 같이 행렬과 데이터 프레임으로 구성되어 있는 리스트 lst를 생성한다.
lst <- list(
  mat = matrix(c(1.2, 2.5, 3.1, 1.5, 2.7, 3.2, 2.1, 2.1, 2.8), nrow = 3),
  df = data.frame(
    x1 = c('Park', 'Lee', 'Kim'),
    x2 = c(14, 16, 21)
  )
)
lst

# 리스트 lst의 각 요소에 다음과 같이 이름을 부여한다.
dimnames(lst$mat) <- list(
  c('Sub1', 'Sub2', 'Sub3'),
  c("Trt1", "Trt2", "Trt3")
)
colnames(lst$df) <- c("name", "sales")
lst


# Girth 값이 평균 이상이고, Height 값이 평균 미만인 데이터를 선택하여, trees_sub1에 저장. 변수는 Girth, Height 만 선택.
trees_sub1 <- trees %>% 
  filter(Girth >= mean(Girth), 
         Height < mean(Height)) %>% 
  select(Girth, Height)
trees_sub1

# Girth 값이 평균 미만이고, Height 값이 평균 이상인 데이터를 선택하여, trees_sub2에 저장. 변수는 Girth, Height 만 선택.
trees_sub2 <- trees %>% 
  filter(Girth < mean(Girth), 
         Height >= mean(Height)) %>% 
  select(Girth, Height)
trees_sub2

# trees_sub1과 trees_sub2의 Girth와 Height의 평균값 및 케이스 값 계산.
trees_sub1 %>%
  summarise(n = n(),
            avg_Girth = mean(Girth),
            avg_Height = mean(Height))
trees_sub2 %>%
  summarise(n = n(),
            avg_Girth = mean(Girth),
            avg_Height = mean(Height))

# mtcars 데이터에서 disp는 세제곱인치 단위의 배기량이므로 cc 단위의 배기량으로 변환 (1세제곱인치 = 16.44cc)하여 disp_cc 생성, cyl에 따라 구분되는 자동차 대수, mpg, disp_cc, hp, wt의 평균값 출력.
mtcars %>% 
  mutate(disp_cc = disp * 16.44) %>% 
  group_by(cyl) %>% 
  summarise(n = n(),
            avg_mpg = mean(mpg),
            avg_disp_cc = mean(disp_cc),
            avg_hp = mean(hp),
            avg_wt = mean(wt))


# 다음의 두 데이터 프레임 (tibble) 생성
part_df <- tibble(num = c(155, 501, 244, 796),
                  tool = c("screwdrive", "pliers", "wrench", "hammer"))
part_df

order_df <- tibble(num = c(155, 796, 155, 244, 244, 796, 244),
                   name = c("Par", "Fox", "Smith", "White", "Crus", "White", "Lee"))
order_df

# 다음의 결과가 되도록 데이터 프레임 결함
part_df %>% left_join(order_df, by = "num")
part_df %>% inner_join(order_df, by = "num")
