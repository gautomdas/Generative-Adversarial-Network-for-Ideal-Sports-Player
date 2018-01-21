require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)

football <- read_csv("football.csv")
football$hw <- NA

scrape <- function(Player, Pos) {
  playerid <- strsplit(Player, "\\", fixed = T)[[1]][2]
  Player <- strsplit(Player, "\\", fixed = T)[[1]][1]
  site <- read_html(paste("https://www.pro-football-reference.com/players/", str_to_upper(substr(playerid, 0, 1)), "/", playerid, ".htm", sep='')) %>%
    html_node("#info") %>%
    html_text()
  hwSpaces <- strsplit(strsplit(strsplit(site, "Position:")[[1]][2], "lb")[[1]][1], "\n\t\n\n\n")[[1]][2]
  hw <- strsplit(hwSpaces, ",")[[1]]
  hw[1] <- as.numeric(strsplit(hw[1], "-")[[1]][1])*12+as.numeric(strsplit(hw[1], "-")[[1]][2])
  hw[2] <- as.numeric(str_trim(hw[2]))
  hw[3] <- playerid
  hw <- paste(hw[1], hw[2], hw[3], sep=",")
  return(hw)
}
baselines <- c(football[football$PosRank == 12 & football$FantPos == 'QB',]$FantPt,
               football[football$PosRank == 24 & football$FantPos == 'RB',]$FantPt,
               football[football$PosRank == 30 & football$FantPos == 'WR',]$FantPt,
               football[football$PosRank == 12 & football$FantPos == 'TE',]$FantPt) #12QB, 24RB, 30WR, 12TE
positionNumbers <- c('QB', 'RB', 'WR', 'TE')

start_time <- Sys.time()
df <- football[1:10,]
# df <- df %>%
football <- football %>%
  filter(!is.na(FantPt)) %>%
  rowwise() %>%
  mutate(hw = scrape(Player, FantPos), VBD = FantPt - baselines[which(positionNumbers == FantPos)]) %>%
  separate(hw, c("height", "weight", "playerid"), ",", convert = T)

end_time <- Sys.time()
show(end_time-start_time)

save(football, file = "football.RData")