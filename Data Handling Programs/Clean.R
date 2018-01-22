require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)

baseball <- function(filename) {
  df <- read_csv(filename)
  df$height <- df$weight <- NA
  df$Name <- df$Team <- df$PA <- df$IP <- df$`*` <- NULL
  
  for (i in 1:nrow(df)) {
    site <- read_html(paste("http://www.fangraphs.com/statss.aspx?playerid=", df$playerid[i], "&position=", df$Pos[i], sep='')) %>%
      html_node(".spacer_10+ div") %>%
      html_text()
    hwSpaces <- strsplit(strsplit(site, "Height/Weight: ")[[1]][2], "Position:")[[1]][1]
    hw <- strsplit(substr(hwSpaces, 0, nchar(hwSpaces)-5), "/")[[1]]
    df$height[i] <- as.numeric(strsplit(hw[1], "-")[[1]][1])*12+as.numeric(strsplit(hw[1], "-")[[1]][2])
    df$weight[i] <- as.numeric(hw[2])
  }
  return(df)
}

basketball <- function(filename) {
  df <- read_csv(filename)
  df <- df[,c(1, 2, 3, 4, 8)]
  df$playerid <- df$height <- df$weight <- NA
  
  for (i in 1:nrow(df)) {
    df$playerid[i] <- strsplit(df$Player[i], "\\", fixed = T)[[1]][2]
    df$Player[i] <- strsplit(df$Player[i], "\\", fixed = T)[[1]][1]
    site <- read_html(paste("https://www.basketball-reference.com/players/", str_to_lower(substr(df$playerid[i], 0, 1)), "/", df$playerid[i], ".html", sep='')) %>%
      html_node("#info") %>%
      html_text()
    hwSpaces <- strsplit(strsplit(strsplit(site, "Shoots:")[[1]][2], "lb")[[1]][1], "\n\n\n\n  ")[[1]][2]
    hw <- strsplit(hwSpaces, ",")[[1]]
    df$height[i] <- as.numeric(strsplit(hw[1], "-")[[1]][1])*12+as.numeric(strsplit(hw[1], "-")[[1]][2])
    df$weight[i] <- as.numeric(str_trim(hw[2]))
  }
  return(df)
}
start_time <- Sys.time()
write_csv(baseball("baseball.csv"), "baseballClean.csv")
end_time <- Sys.time()
show(end_time-start_time)

start_time <- Sys.time()
write_csv(basketball("basketball.csv"), "basketballClean.csv")
end_time <- Sys.time()
show(end_time-start_time)