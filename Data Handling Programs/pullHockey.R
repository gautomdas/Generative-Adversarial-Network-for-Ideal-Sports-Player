require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)

hockey <- read_csv("hockey.csv")
hockey$hw <- NA

scrape <- function(Player, Pos) {
  playerid <- strsplit(Player, "\\", fixed = T)[[1]][2]
  Player <- strsplit(Player, "\\", fixed = T)[[1]][1]
  site <- read_html(paste("https://www.hockey-reference.com/players/", str_to_lower(substr(playerid, 0, 1)), "/", playerid, ".html", sep='')) %>%
    html_node("#info") %>%
    html_text()
  hwSpaces <- strsplit(strsplit(strsplit(site, "Shoots:")[[1]][2], "lb")[[1]][1], "\n\n\n\n")[[1]][2]
  hw <- strsplit(hwSpaces, ",")[[1]]
  hw[1] <- as.numeric(strsplit(hw[1], "-")[[1]][1])*12+as.numeric(strsplit(hw[1], "-")[[1]][2])
  hw[2] <- as.numeric(str_trim(hw[2]))
  hw[3] <- playerid
  hw <- paste(hw[1], hw[2], hw[3], sep=",")
  return(hw)
}
start_time <- Sys.time()
hockey <- hockey %>%
  filter(GP >= 20) %>%
  rowwise() %>%
  mutate(hw = scrape(Player, Pos)) %>%
  separate(hw, c("height", "weight", "playerid"), ",", convert = T)

end_time <- Sys.time()
show(end_time-start_time)

save(hockey, file = "hockey.RData")