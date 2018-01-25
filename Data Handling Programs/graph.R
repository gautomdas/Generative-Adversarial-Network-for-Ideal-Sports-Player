require(ggplot2)
require(readr)

df <- read_csv("baseballClean.csv")
names(df)[names(df) == "label"] <- "rank"
df$distFromMiddle <- dist(df$z_score, mean(df$z_score))

df$clusters <- kmeans(df$distFromMiddle, 10)

p <- ggplot(df, aes(BMI, z_score)) + geom_point() + geom_smooth(method = "lm")
print(p)
