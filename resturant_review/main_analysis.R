# Load required libraries
library(readr)
library(dplyr)
library(tm)
library(caret)
library(ggplot2)
library(wordcloud2)

# Clear workspace and garbage collect to free memory
rm(list = ls())
gc()

# Read the dataset with proper encoding
Data <- read_csv("converttsv-Resturant-R.csv", locale = locale(encoding = "Latin1"))

# Remove duplicate rows
Data <- distinct(Data)

# Fix character encoding issues
Data$Review <- iconv(Data$Review, from = "latin1", to = "UTF-8", sub = "")

# Text preprocessing function
text_preprocessing <- function(x) {
  x <- tolower(x)
  x <- gsub('[[:cntrl:][:punct:]]', '', x)
  x <- gsub('\\d+', '', x)
  x <- gsub('[[:space:]]+', ' ', x)
  x <- removeWords(x, stopwords("english"))
  return(trimws(x))
}

# Apply text preprocessing
Data$Review <- sapply(Data$Review, text_preprocessing)

# Remove empty reviews
Data <- Data[Data$Review != "", ]

# Create a text corpus
corpus <- Corpus(VectorSource(Data$Review))

# Generate the term-document matrix
tdm <- TermDocumentMatrix(corpus, control = list(wordLengths = c(1, Inf)))

# Adjust sparsity threshold to keep more terms
tdm <- removeSparseTerms(tdm, sparse = 0.99)  # Keep more terms than before

# Convert to TF-IDF matrix and transpose it
tfidf <- weightTfIdf(tdm)
tfidf_df <- as.data.frame(t(as.matrix(tfidf)))
colnames(tfidf_df) <- make.names(colnames(tfidf_df))

# Ensure that the number of rows matches before combining with the target
if (nrow(tfidf_df) != nrow(Data)) {
  stop(paste("Row mismatch:", nrow(tfidf_df), "vs", nrow(Data)))
}
tfidf_df$Liked <- as.factor(Data$Liked)

# Split data into training and testing sets
set.seed(123)
split <- sample(1:nrow(tfidf_df), 0.8 * nrow(tfidf_df))
train <- tfidf_df[split, ]
test <- tfidf_df[-split, ]

# Train logistic regression model (can take time if matrix is still large)
system.time({
  model <- glm(Liked ~ ., data = train, family = "binomial")
})

# Predict on test set
preds <- predict(model, newdata = test, type = "response")
pred_classes <- ifelse(preds > 0.5, 1, 0)

# Evaluate with confusion matrix
confusionMatrix(factor(pred_classes, levels = c(0, 1)), test$Liked)

# Word frequency analysis (top 25 most frequent words)
term_freq <- rowSums(as.matrix(tdm))
df <- data.frame(term = names(term_freq), freq = term_freq)
top_terms <- df %>% arrange(desc(freq)) %>% head(25)

# Bar chart of top 25 most frequent terms
ggplot(top_terms, aes(x = reorder(term, freq), y = freq)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 25 Frequent Words", x = "Word", y = "Frequency")

# Generate a word cloud
wordcloud2(top_terms, color = "random-dark", backgroundColor = "white")
