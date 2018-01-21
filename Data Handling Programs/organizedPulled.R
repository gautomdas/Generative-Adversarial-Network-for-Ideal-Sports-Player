require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)

load("basketball.RData")
load("baseball.RData")
load("hockey.RData")

baseball <- baseball %>%
  select(Name, playerid, Team, Pos, WAR, height, weight) %>%
  mutate(z_score = (WAR-mean(baseball$WAR))/sd(baseball$WAR)) %>%
  arrange(desc(z_score), height, weight)

basketball <- basketball %>%
  rename(Name = Player, Team = Tm) %>%
  select(Name, playerid, Team, Pos, PER, height, weight) %>%
  mutate(Name = strsplit(Name, "\\", fixed = T)[[1]][1], z_score = (PER-mean(basketball$PER))/sd(basketball$PER)) %>%
  arrange(desc(z_score), height, weight)

hockey <- hockey %>%
  rename(Name = Player, Team = Tm, ExpectedPM = `E+/-`)
hockey <- hockey %>% #we have to do it twice because of some weird dplyr stuff where the name change isn't immediately recognized or something like that
  select(Name, playerid, Team, Pos, ExpectedPM, height, weight) %>%
  mutate(Name = strsplit(Name, "\\", fixed = T)[[1]][1], z_score = (ExpectedPM-mean(hockey$ExpectedPM))/sd(hockey$ExpectedPM)) %>%
  arrange(desc(z_score), height, weight)

write_csv(baseball, "baseballClean.csv")
write_csv(basketball, "basketballClean.csv")
write_csv(hockey, "hockeyClean.csv")