library(tidyverse)
library(ggpubr)
library(caret)
library(scales)
library(xlsx)

#import data and replace empty cells with NA
#move target variable "Converted" to first column
leads <- read.csv("Leads X Education.csv", na.strings = c(""," ", "NA")) %>%
  relocate(Converted, .before = Prospect.ID)

#view head and tail of data
head(leads)
tail(leads)

#view structure and summary of data
str(leads)
summary(leads)

#examine options in each option set column
table(leads$Converted)
table(leads$Lead.Origin)
table(leads$Lead.Source)
table(leads$Last.Activity)
table(leads$Last.Notable.Activity)
table(leads$Specialization)
table(leads$How.did.you.hear.about.X.Education)
table(leads$What.is.your.current.occupation)
table(leads$Tags)
table(leads$Lead.Quality)
table(leads$What.matters.most.to.you.in.choosing.a.course)
table(leads$Lead.Profile)

#replace column levels of Select with NA
leads$Specialization[leads$Specialization == "Select"] <- NA
leads$How.did.you.hear.about.X.Education[leads$How.did.you.hear.about.X.Education == "Select"] <- NA
leads$Lead.Profile[leads$Lead.Profile == "Select"] <- NA

#replace NA values with NULL
leads$Lead.Origin[is.na(leads$Lead.Origin)] <- "NULL"
leads$Lead.Source[is.na(leads$Lead.Source)] <- "NULL"
leads$Last.Activity[is.na(leads$Last.Activity)] <- "NULL"
leads$Last.Notable.Activity[is.na(leads$Last.Notable.Activity)] <- "NULL"
leads$Specialization[is.na(leads$Specialization)] <- "NULL"
leads$How.did.you.hear.about.X.Education[is.na(leads$How.did.you.hear.about.X.Education)] <- "NULL"
leads$What.is.your.current.occupation[is.na(leads$What.is.your.current.occupation)] <- "NULL"
leads$What.matters.most.to.you.in.choosing.a.course[is.na(leads$What.matters.most.to.you.in.choosing.a.course)] <- "NULL"
leads$Lead.Profile[is.na(leads$Lead.Profile)] <- "NULL"

#convert yes/no columns to binary using 
leads$Do.Not.Call <- as.integer(ifelse(leads$Do.Not.Call == "Yes",1,0))
leads$Do.Not.Email <- as.integer(ifelse(leads$Do.Not.Email == "Yes",1,0))
leads$Search <- as.integer(ifelse(leads$Search == "Yes",1,0))
leads$Magazine <- as.integer(ifelse(leads$Magazine == "Yes",1,0))
leads$Newspaper.Article <- as.integer(ifelse(leads$Newspaper.Article == "Yes",1,0))
leads$X.Education.Forums <- as.integer(ifelse(leads$X.Education.Forums == "Yes",1,0))
leads$Newspaper <- as.integer(ifelse(leads$Newspaper == "Yes",1,0))
leads$Digital.Advertisement <- as.integer(ifelse(leads$Digital.Advertisement == "Yes",1,0))
leads$Through.Recommendations <- as.integer(ifelse(leads$Through.Recommendations == "Yes",1,0))
leads$Receive.More.Updates.About.Our.Courses <- as.integer(ifelse(leads$Receive.More.Updates.About.Our.Courses == "Yes",1,0))
leads$Update.me.on.Supply.Chain.Content <- as.integer(ifelse(leads$Update.me.on.Supply.Chain.Content == "Yes",1,0))
leads$Get.updates.on.DM.Content <- as.integer(ifelse(leads$Get.updates.on.DM.Content == "Yes",1,0))
leads$I.agree.to.pay.the.amount.through.cheque <- as.integer(ifelse(leads$I.agree.to.pay.the.amount.through.cheque == "Yes",1,0))
leads$A.free.copy.of.Mastering.The.Interview <- as.integer(ifelse(leads$A.free.copy.of.Mastering.The.Interview == "Yes",1,0))
str(leads)

#convert columns to factor for one-hot encoding
leads$Lead.Origin <- as.factor(leads$Lead.Origin)
leads$Lead.Source <- as.factor(leads$Lead.Source)
leads$Last.Activity <- as.factor(leads$Last.Activity)
leads$Last.Notable.Activity <- as.factor(leads$Last.Notable.Activity)
leads$Specialization <- as.factor(leads$Specialization)
leads$How.did.you.hear.about.X.Education <- as.factor(leads$How.did.you.hear.about.X.Education)
leads$What.is.your.current.occupation <- as.factor(leads$What.is.your.current.occupation)
leads$What.matters.most.to.you.in.choosing.a.course <- as.factor(leads$What.matters.most.to.you.in.choosing.a.course)
leads$Lead.Profile <- as.factor(leads$Lead.Profile)

#drop unwanted columns
#removing lead quality because score based on employee intuition
sub1 = subset(leads, select = -c(Prospect.ID, Lead.Number, Country, City, Tags, Lead.Quality))
summary(sub1)
#one hot encode features with more than 2 levels
#https://www.pluralsight.com/guides/encoding-data-with-r
#https://www.r-bloggers.com/2020/02/a-guide-to-encoding-categorical-features-using-r/
leads_enc <- sub1
dmy <- dummyVars("~ .", sep=".", data = leads_enc)
leads_clean <- data.frame(predict(dmy, newdata = leads_enc))
str(leads_clean)
head(leads_clean)

#convert Converted column to factor
sub1$Converted <- as.factor(sub1$Converted)

#convert Converted column to text factor
leads_clean$Converted <- as.factor(ifelse(leads_clean$Converted == 0, "No", "Yes"))

#drop NULL columns
clean = subset(leads_clean, select = -c(Lead.Source.NULL, Last.Activity.NULL, 
                                        Specialization.NULL, How.did.you.hear.about.X.Education.NULL, 
                                        What.is.your.current.occupation.NULL, What.matters.most.to.you.in.choosing.a.course.NULL,
                                        Lead.Profile.NULL))

#create list of na value counts for each column
#https://stackoverflow.com/questions/24027605/determine-the-number-of-na-values-in-a-column
na_count <- sapply(clean, function(y) sum(length(which(is.na(y)))))

#convert list to data frame and print results
na_count <- data.frame(na_count)
view(na_count)

#remove rows with na values
sub2 <- na.omit(clean)
str(sub2)

#EDA plots
#0 = no and 1 = yes

#define function to plot small integers with pretty_breaks()
#https://www.r-bloggers.com/2019/11/setting-axes-to-integer-values-in-ggplot2/
int_breaks <- function(n = 5, ...) {
  fxn <- function(x) {
    breaks <- floor(pretty(x, n, ...))
    names(breaks) <- attr(breaks, "labels")
    breaks
  }
  return(fxn)
}

#plot do not call to test integer breaks
ggplot(leads, aes(x=Do.Not.Call)) +
  geom_bar() +
  scale_x_continuous(breaks = int_breaks())

#count of converted leads
ggplot(sub1, aes(x=Converted)) +
  geom_bar() +
  ggtitle("Converted Leads") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank())

#chart distribution of ads

search <- ggplot(sub1, aes(x=Search)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..)) +
  ggtitle("Ad through Search") +
  scale_x_continuous(breaks = int_breaks())

magazine <- ggplot(sub1, aes(x=Magazine)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..)) +
  ggtitle("Ad in Magazine") +
  scale_x_continuous(breaks = int_breaks())

newspaper <- ggplot(sub1, aes(x=Newspaper.Article)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..)) +
  ggtitle("Ad in Newspaper") +
  scale_x_continuous(breaks = int_breaks())

forum <- ggplot(sub1, aes(x=X.Education.Forums)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..)) +
  ggtitle("Ad through X Education Forums") +
  scale_x_continuous(breaks = int_breaks())

digital <- ggplot(sub1, aes(x=Digital.Advertisement)) +
  geom_bar() +
  geom_text(stat='count', aes(label=..count..)) +
  ggtitle("Ad in Digital Ad") +
  scale_x_continuous(breaks = int_breaks())

ggarrange(search + theme(axis.title.x = element_blank(), axis.title.y = element_blank()),
          magazine + theme(axis.title.x = element_blank(), axis.title.y = element_blank()),
          newspaper + theme(axis.title.x = element_blank(), axis.title.y = element_blank()), 
          forum + theme(axis.title.x = element_blank(), axis.title.y = element_blank()),
          digital + theme(axis.title.x = element_blank(), axis.title.y = element_blank()),
          ncol = 3, nrow = 2)
  
#combine boxplots

#converted leads by time on website
conv.time <- ggplot(sub1, aes(x=Converted, y=Total.Time.Spent.on.Website)) +
  geom_boxplot() +
  ggtitle("Converted Leads by Time on Website")

#converted leads by page views
conv.views <- ggplot(sub1, aes(x=Converted, y=Page.Views.Per.Visit)) +
  geom_boxplot() +
  ggtitle("Converted Leads by Page Views")

#lead origin by time on website
origin.time <- ggplot(sub1, aes(x=Lead.Origin, y=Total.Time.Spent.on.Website)) +
  geom_boxplot() +
  ggtitle("Lead Origin by Time on Website")

#lead origin by page views
origin.views <- ggplot(sub1, aes(x=Lead.Origin, y=Page.Views.Per.Visit)) +
  geom_boxplot() +
  ggtitle("Lead Origin by Page Views")

ggarrange(conv.time + theme(axis.title.x = element_blank(), axis.title.y = element_blank()),
          conv.views + theme(axis.title.x = element_blank(), axis.title.y = element_blank()),
          origin.time + theme(axis.text.x = element_text(angle=45, hjust = 1), 
                              axis.title.x = element_blank(), axis.title.y = element_blank()), 
          origin.views + theme(axis.text.x = element_text(angle=45, hjust = 1), 
                               axis.title.x = element_blank(), axis.title.y = element_blank()),
          ncol = 2, nrow = 2)

#combine 4 barplots

#sort by most frequent
#fct_infreq from tidyverse package
origin <- ggplot(sub1, aes(x=fct_infreq(Lead.Origin))) +
  geom_bar() +
  ggtitle("Count of Lead Origin")

source <- ggplot(sub1, aes(x=fct_infreq(Lead.Source))) +
  geom_bar() +
  ggtitle("Count of Lead Source")

updates <- ggplot(sub1, aes(x=Receive.More.Updates.About.Our.Courses)) +
  geom_bar() +
  ggtitle("Count of Receive More Updates") +
  scale_x_continuous(breaks = int_breaks())

free <- ggplot(sub1, aes(x=A.free.copy.of.Mastering.The.Interview)) +
  geom_bar() +
  ggtitle("Count of Free Incentive") +
  scale_x_continuous(breaks = int_breaks())

ggarrange(origin + theme(axis.text.x = element_text(angle=45, hjust = 1), 
                         axis.title.x = element_blank(), axis.title.y = element_blank()), 
          source + theme(axis.text.x = element_text(angle=45, hjust = 1), 
                         axis.title.x = element_blank(), axis.title.y = element_blank()), 
          updates + theme(axis.title.x = element_blank(), axis.title.y = element_blank()), 
          free + theme (axis.title.x = element_blank(), axis.title.y = element_blank()),
          ncol = 2, nrow = 2)

#histogram of time spent on website
ggplot(data = sub1, aes(x=Total.Time.Spent.on.Website)) +
  geom_histogram(bins = 20) +
  ggtitle("Histogram of Time Spent on Website")

#histogram of page views
ggplot(data = sub1, aes(x=Page.Views.Per.Visit)) +
  geom_histogram() +
  ggtitle("Histogram of Page Views")

#histograms of asymmetric scores
ggplot(data = sub1, aes(x = Asymmetrique.Activity.Score)) +
  geom_bar() +
  ggtitle("Count of Asymmetric Activity Score")

ggplot(data = sub1, aes(x = Asymmetrique.Profile.Score)) +
  geom_bar() +
  ggtitle("Count of Asymmetric Profile Score")

#plot lead origin
ggplot(data = sub1, aes(x = Lead.Origin)) +
  geom_bar() +
  ggtitle("Lead Origin")

#plot what matters most
ggplot(data = sub1, aes(x = What.matters.most.to.you.in.choosing.a.course)) +
  geom_bar() +
  ggtitle("What Matters Most")

#plot occupation
ggplot(data = sub1, aes(x = What.is.your.current.occupation)) +
  geom_bar() +
  ggtitle("Current Occupation")

#plot Page Views boxplot and remove outliers

ggplot(data = sub1, aes(x=Page.Views.Per.Visit)) +
  geom_boxplot() +
  ggtitle("Page Views")

#extract Page Views outliers
out <- boxplot.stats(sub1$Page.Views.Per.Visit)$out
#identify outlier rows
out_ind <- which(sub1$Page.Views.Per.Visit %in% c(out))
out_ind
#remove from dataset
clean = clean[-which(clean$Page.Views.Per.Visit %in% out),]
#plot Page Views
ggplot(data = clean, aes(x=Page.Views.Per.Visit)) +
  geom_boxplot() +
  ggtitle("Page Views - Clean")
#write final clean data to local file
write.csv(clean, "clean.csv")