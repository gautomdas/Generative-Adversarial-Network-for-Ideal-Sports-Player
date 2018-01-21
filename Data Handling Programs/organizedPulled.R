require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)

load("basketball.RData")
load("baseball.RData")

baseball <- baseball %>%
  select(Name, playerid, Team, Pos, WAR, height, weight) %>%
  mutate(z_score = (WAR-mean(baseball$WAR))/sd(baseball$WAR)) %>%
  arrange(desc(z_score), height, weight)

basketball <- basketball %>%
  rename(Name = Player, Team = Tm) %>%
  select(Name, playerid, Team, Pos, PER, height, weight) %>%
  mutate(Name = strsplit(Name, "\\", fixed = T)[[1]][1], z_score = (PER-mean(basketball$PER))/sd(basketball$PER)) %>%
  arrange(desc(z_score), height, weight)

write_csv(baseball, "baseballClean.csv")
write_csv(basketball, "basketballClean.csv")