require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)

baseball <- read_csv("baseball.csv")
baseball$hw <- NA

scrape <- function(playerid, Pos) {
  site <- read_html(paste("http://www.fangraphs.com/statss.aspx?playerid=", playerid, "&position=", Pos, sep='')) %>%
    html_node(".spacer_10+ div") %>%
    html_text()
  hwSpaces <- strsplit(strsplit(site, "Height/Weight: ")[[1]][2], "Position:")[[1]][1]
  hw <- strsplit(substr(hwSpaces, 0, nchar(hwSpaces)-5), "/")[[1]]
  hw[1] <- as.numeric(strsplit(hw[1], "-")[[1]][1])*12+as.numeric(strsplit(hw[1], "-")[[1]][2])
  hw[2] <- as.numeric(hw[2])
  hw <- paste(hw[1], hw[2], sep=",")
  return(hw)
}

start_time <- Sys.time()

baseball <- baseball %>%
  filter(PA >= 200 | IP >= 50) %>%
  rowwise() %>%
  mutate(hw = scrape(playerid, Pos)) %>%
  separate(hw, c("height", "weight"), ",", convert = T)

end_time <- Sys.time()
show(end_time-start_time)

save(baseball, file = "baseball.RData")