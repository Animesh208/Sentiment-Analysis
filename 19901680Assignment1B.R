library(syuzhet)
library(tidytext)

library(textdata)
library(dplyr)
library(ggplot2)
library(tm)
library(caret)
library(e1071)
speech01 <- read.csv(file = "hotel_tweets.csv")
nrc_lexicon <- get_sentiments("nrc")
speech <- paste(speech01$Negative,speech01$Positive)
nrcData <- get_nrc_sentiment(speech)
## We	will	next	analyse	the	speech	in	terms	of	the	8	emotions.
td <- data.frame(t(nrcData))
tdSum <- data.frame(rowSums(td))
names(tdSum)[1] <- "Count"
tdSum <- cbind("sentiment" = rownames(tdSum), tdSum)
rownames(tdSum) <- NULL
tdSum <- tdSum[1:8,]

ggplot(data=tdSum, aes(x = sentiment)) +
  geom_bar(aes(weight=Count, fill=sentiment)) +
  ggtitle("Speech Sentiments") + guides(fill=FALSE)

sentences <- get_sentences(speech)
sentences_df <- tibble(sentence = 1:length(sentences),content = sentences)
sentiment <- get_sentiment(sentences)

tidy_df<-unnest_tokens(sentences_df, output=word, input=content)
tidy_df

## JOY
nrc_joy <- get_sentiments("nrc") %>% filter(sentiment == "joy")


joy_words <- tidy_df %>% inner_join(nrc_joy) %>% count(word, sort = TRUE)

joy_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "orange", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("Joy Emotion")

## anticipation
nrc_anticipation <- get_sentiments("nrc") %>% filter(sentiment == "anticipation")


anticipation_words <- tidy_df %>% inner_join(nrc_anticipation) %>% count(word, sort = TRUE)

anticipation_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "red", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("anticipation Emotion")

## disgust
nrc_disgust <- get_sentiments("nrc") %>% filter(sentiment == "disgust")

disgust_words <- tidy_df %>% inner_join(nrc_disgust) %>% count(word, sort = TRUE)


disgust_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "red", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("disgust Emotion")


## anger
nrc_anger <- get_sentiments("nrc") %>% filter(sentiment == "anger")


anger_words <- tidy_df %>% inner_join(nrc_anger) %>% count(word, sort = TRUE)

anger_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "red", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("anger Emotion")


## fear
nrc_fear <- get_sentiments("nrc") %>% filter(sentiment == "fear")

fear_words <- tidy_df %>% inner_join(nrc_fear) %>% count(word, sort = TRUE)


fear_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "red", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("fear Emotion")

## sadness
nrc_sadness <- get_sentiments("nrc") %>% filter(sentiment == "sadness")

sadness_words <- tidy_df %>% inner_join(nrc_sadness) %>% count(word, sort = TRUE)


sadness_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "red", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("sadness Emotion")


## surprise
nrc_surprise <- get_sentiments("nrc") %>% filter(sentiment == "surprise")

surprise_words <- tidy_df %>% inner_join(nrc_surprise) %>% count(word, sort = TRUE)


surprise_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "blue", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("surprise Emotion")

## trust
nrc_trust <- get_sentiments("nrc") %>% filter(sentiment == "trust")

trust_words <- tidy_df %>% inner_join(nrc_trust) %>% count(word, sort = TRUE)


trust_words %>%
  head(5) %>%
  ggplot(aes(reorder(word, n), n)) +
  geom_bar(alpha = 0.8, fill = "blue", stat = "identity") +
  geom_text(aes(label=n), hjust = -0.3, size=3.5) +
  labs(y = "Contribution to sentiment", x = NULL) +
  coord_flip() +
  ggtitle("trust Emotion")





##COLUMN 1 = NEGATIVE, COLUMN2 = POSITIVE
Train <- c(as.character(speech01[1:200,1]),as.character(speech01[1:200,2]))
Test <- c(as.character(speech01[201:263,1]), as.character(speech01[201:263,2]))
all <- c(Train, Test)

View(Train)

sentimentTrain <- c(rep("negative", length(speech01[1:200,1])), rep("positive",length(speech01[1:200,2])))
sentimentTest <- c(rep("negative", length(speech01[201:263,1])), rep("positive",length(speech01[201:263,2])))
sentimentAll <- as.factor(c(sentimentTrain, sentimentTest))

tweetsCorpus <- Corpus(VectorSource(all))

tweetsCorpus <- tm_map(tweetsCorpus, removeNumbers)
tweetsCorpus <- tm_map(tweetsCorpus, stripWhitespace)
tweetsCorpus <- tm_map(tweetsCorpus, content_transformer(tolower))
tweetsCorpus <- tm_map(tweetsCorpus, removeWords, stopwords("english"))

tweetsDTM <- DocumentTermMatrix(tweetsCorpus)

tweetsMatrix <- as.matrix(tweetsDTM)

svmClassifier <- svm(tweetsMatrix[1:400,], as.factor(sentimentAll[1:400]))

svmPredicted <- predict(svmClassifier, tweetsMatrix[401:526,])
svmPredicted

table(svmPredicted, sentimentTest)
confusionMatrix(as.factor(svmPredicted), as.factor(sentimentTest))

