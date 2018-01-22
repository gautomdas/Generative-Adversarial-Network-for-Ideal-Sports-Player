require(dplyr)
require(tidyr)
require(readr)
require(rvest)
require(stringr)
require(scales)

load("basketball.RData")
load("baseball.RData")
load("hockey.RData")
load("football.RData")

baseball <- baseball %>%
  filter(!is.na(height) & !is.na(weight)) %>%
  select(Name, playerid, Team, Pos, WAR, height, weight) %>%
  mutate(z_score = (WAR-mean(baseball$WAR))/sd(baseball$WAR), BMI = 703*weight/(height^2)) %>%
  arrange(desc(z_score), height, weight)
baseball <- baseball %>%
  mutate(combined_data = paste(z_score, height, weight, BMI, sep = "_"))
baseball <- baseball %>%
  mutate(label = as.integer((z_score-min(baseball$z_score))/(max(baseball$z_score)-min(baseball$z_score))*10))

basketball <- basketball %>%
  filter(!is.na(height) & !is.na(weight)) %>%
  rename(Name = Player, Team = Tm) %>%
  select(Name, playerid, Team, Pos, PER, height, weight) %>%
  mutate(Name = strsplit(Name, "\\", fixed = T)[[1]][1], z_score = (PER-mean(basketball$PER))/sd(basketball$PER), BMI = 703*weight/(height^2)) %>%
  arrange(desc(z_score), height, weight)
basketball <- basketball %>%
  mutate(combined_data = paste(z_score, height, weight, BMI, sep = "_"))
basketball <- basketball %>%
  mutate(label = as.integer((z_score-min(basketball$z_score))/(max(basketball$z_score)-min(basketball$z_score))*10))

hockey <- hockey %>%
  filter(!is.na(height) & !is.na(weight)) %>%
  rename(Name = Player, Team = Tm, ExpectedPM = `E+/-`)
hockey <- hockey %>% #we have to do it twice because of some weird dplyr stuff where the name change isn't immediately recognized or something like that
  select(Name, playerid, Team, Pos, ExpectedPM, height, weight) %>%
  mutate(Name = strsplit(Name, "\\", fixed = T)[[1]][1], z_score = (ExpectedPM-mean(hockey$ExpectedPM))/sd(hockey$ExpectedPM), BMI = 703*weight/(height^2)) %>%
  arrange(desc(z_score), height, weight)
hockey <- hockey %>%
  mutate(combined_data = paste(z_score, height, weight, BMI, sep = "_"))
hockey <- hockey %>%
  mutate(label = as.integer((z_score-min(hockey$z_score))/(max(hockey$z_score)-min(hockey$z_score))*10))

football <- football %>%
  filter(!is.na(height) & !is.na(weight)) %>%
  rename(Name = Player, Team = Tm, Pos = FantPos) %>%
  select(Name, playerid, Team, Pos, VBD, height, weight) %>%
  mutate(Name = strsplit(Name, "\\", fixed = T)[[1]][1], z_score = (VBD-mean(football$VBD))/sd(football$VBD), BMI = 703*weight/(height^2)) %>%
  arrange(desc(z_score), height, weight)
football <- football %>%
  mutate(combined_data = paste(z_score, height, weight, BMI, sep = "_"))
football <- football %>%
  mutate(label = as.integer((z_score-min(football$z_score))/(max(football$z_score)-min(football$z_score))*10))

write_csv(baseball, "baseballClean.csv")
write_csv(basketball, "basketballClean.csv")
write_csv(hockey, "hockeyClean.csv")
write_csv(football, "footballClean.csv")