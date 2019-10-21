x <- c(3.5, 3.5, 3.2, 3.0, 3.1, 3.3, 3  , 3.5, 3.5, 3.9)
y <- c(3.4, 3.5, 3.2, 3.0, 3.0, 3.3, 3.1, 3.5, 3.6, 3.9)
plot(x, y, pch = 19)
mean(x)
mean(y)
x1 <- x - mean(x)
x1
summary(x1)
y1 <- y - mean(y)
y1
summary(y1)
plot(x1, y1, pch = 19)
data <- data.frame(x,y)
data.pca <- prcomp(data)

data.pca
summary(data.pca)
plot(data.pca$x[,1], data.pca$x[,2])

