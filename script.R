################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unwanted objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Create test and train sets from edx set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# Remove unwanted objects
rm(test_index, temp, removed)

# Create rmse function to easily calculate the rmse for each model
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

##################################################################################### Data Exploration

str(edx)

head(edx)

summary(edx$rating)

edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

edx %>% summarize(n_genres = n_distinct(genres))

# Top 10 genres
edx %>% 
  group_by(genres) %>% 
  summarize(mean_rating = mean(rating)) %>% 
  top_n(mean_rating, n = 10) %>% 
  arrange(desc(mean_rating))

# Bottom 10 genres
edx %>% 
  group_by(genres) %>% 
  summarize(mean_rating = mean(rating)) %>% 
  top_n(mean_rating, n = -10) %>% 
  arrange(mean_rating)

# Visualize variability of a sample of genres
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 100000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# User rating count distribution
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")

# Mean user rating distribution
edx %>% 
  group_by(userId) %>% summarize(mean_rating = mean(rating)) %>% 
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 30, color = "black")

# Movie rating count distribution
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# Mean movie rating distribution
edx %>%
  group_by(movieId) %>% summarize(mean_rating = mean(rating)) %>% 
  ggplot(aes(mean_rating)) +
  geom_histogram(bins = 30, color = "black")

# Creat small user x movie matrix for illustrative purposes
keep <- edx %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)

tab <- edx %>%
  filter(userId %in% c(13:20)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()

# Create plot of larger sample to visualize how sparse the rating data is
users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

##################################################################################### Naive Model

mu <- mean(train_set$rating)

naive_rmse <- RMSE(test_set$rating, mu)

# Calculate RMSE
rmse_results <- data_frame(method = "Just the average", 
                           RMSE = naive_rmse)

rmse <- round(naive_rmse, 4)

rmse_results %>% knitr::kable() %>% 
  kable_styling(latex_options=c("striped")) %>% 
  column_spec(1:2, color = "black") %>% 
  row_spec(0, color = "white", background = "#5a5cd6")

##################################################################################### Movie Effect Model

# Calculate average rating
mu <- mean(train_set$rating)

# Calculate the movie effect by removing the mean
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# Prediction based on mean plus movie effect
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate RMSE
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))

# Absolute improvement from initial model
imp <- round(naive_rmse - model_1_rmse, 4)
imp

# Percent improvement from initial model
imp_perc <- paste(
  round(imp / naive_rmse, 2)*100, 
  "%", 
  sep = "")
imp_perc

rmse_results %>% knitr::kable() %>% 
  kable_styling(latex_options=c("striped")) %>% 
  column_spec(1:2, color = "black") %>% 
  row_spec(0, color = "white", background = "#5a5cd6")

##################################################################################### Movie + User Effect Model

# Calculate user effects by removing the mean and movie effects
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Prediction based on mean plus movie effect and user effect
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calculate RMSE
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie + User Effect Model",  
                                     RMSE = model_2_rmse ))

# Absolute improvement from initial model
imp <- round(naive_rmse - model_2_rmse, 4)
imp

# Percent improvement from initial model
imp_perc <- paste(
  round(imp / naive_rmse, 2)*100, 
  "%", 
  sep = "")
imp_perc

rmse_results %>% knitr::kable() %>% 
  kable_styling(latex_options=c("striped")) %>% 
  column_spec(1:2, color = "black") %>% 
  row_spec(0, color = "white", background = "#5a5cd6")

##################################################################################### Regularized Movie + User Effect Model

# Create vector of lambdas to use for cross validation
lambdas <- seq(0, 10, 0.25)

# Since lambda is a tuning parameter, we can't use the test set to select it.
# The code is the course CANNOT be used because the selection of lambda was 
# used using the test set. Professor Irizarry even tells us this in the 
# video lecture covering regularization.
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    # This is where the example from the course had the test set called.
    # Train set was substituted to make sure we are not cheating in our
    # selection of lambda.
    train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, train_set$rating))
})

tibble(Lambda = lambdas, RMSE = rmses) %>% 
  ggplot(aes(Lambda, RMSE)) +
  geom_point() +
  geom_text(aes(x = lambdas[which.min(rmses)], 
                y = 0.8565, 
                label = paste("Best Lambda: ", 
                              lambdas[which.min(rmses)]))) +
  geom_vline(xintercept = lambdas[which.min(rmses)], color = "blue")

lambda <- lambdas[which.min(rmses)]

mu <- mean(train_set$rating)

b_i <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

predicted_ratings <- 
  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

# Calculate RMSE
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = model_3_rmse))

# Absolute improvement from initial model
imp <- round(naive_rmse - model_3_rmse, 4)
imp

# Percent improvement from initial model
imp_perc <- paste(
  round(imp / naive_rmse, 2)*100, 
  "%", 
  sep = "")
imp_perc

rmse_results %>% knitr::kable() %>% 
  kable_styling(latex_options=c("striped")) %>% 
  column_spec(1:2, color = "black") %>% 
  row_spec(0, color = "white", background = "#5a5cd6")

##################################################################################### Matrix factorization

# Create small set for illustrating how movie and user effect model 
# doesn't account for all the structure in the data.
train_small <- edx %>% 
  group_by(movieId) %>%
  filter(n() >= 1000) %>% ungroup() %>%
  group_by(userId) %>%
  filter(n() >= 1000) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

movie_titles <- edx %>% 
  select(movieId, title) %>%
  distinct()

rownames(y)<- y[,1]
y <- y[,-1]
colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])

# Create residuals for plotting
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))

# Plot residuals for Toy Story vs Jumanji
m_1 <- "Toy Story (1995)"
m_2 <- "Jumanji (1995)"
qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)

# Plot residuals for Pulp Fiction vs Natural Born Killers
m_3 <- "Pulp Fiction (1994)"
m_4 <- "Natural Born Killers (1994)"
qplot(y[ ,m_3], y[,m_4], xlab = m_3, ylab = m_4)

set.seed(1, sample.kind="Rounding")

# Create train and test, user by movie matrices
train_reco <- with(train_set, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating))

test_reco <- with(test_set, data_memory(user_index = userId,
                                        item_index = movieId,
                                        rating = rating))

# Create reco model object
r <- recosystem::Reco()

# Select optimal tuning parameters
options <- r$tune(train_reco, opts = list(dim = seq(10,30,10),
                                          lrate = c(0.05, 0.1, 0.2),
                                          nthread = 4,
                                          costp_l2 = c(0.01, 0.1),
                                          costq_l2 = c(0.01, 0.1),
                                          niter = 10,
                                          nfold = 10))

# Train reco model
r$train(train_reco, opts = c(options$min,
                             nthread = 4,
                             niter = 20,
                             verbose = FALSE))

# Predict test set using model
predict_reco <-  r$predict(test_reco, out_memory())

# Calculate RMSE
model_4_rmse <- RMSE(predict_reco, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix factorization",  
                                     RMSE = model_4_rmse ))

rmse <- round(model_4_rmse, 2)

# Absolute improvement from initial model
imp <- round(naive_rmse - rmse, 4)

# Percent improvement from initial model
imp_perc <- paste(
  round(imp / naive_rmse, 2)*100, 
  "%", 
  sep = "")

rmse_results %>% knitr::kable() %>% 
  kable_styling(latex_options=c("striped")) %>% 
  column_spec(1:2, color = "black") %>% 
  row_spec(0, color = "white", background = "#5a5cd6")

##################################################################################### Final validation w/ Matrix factorization

set.seed(1, sample.kind="Rounding")

# Create edx and validation, user by movie matrices
edx_reco <- with(edx, data_memory(user_index = userId,
                                  item_index = movieId,
                                  rating = rating))

valid_reco <- with(validation, data_memory(user_index = userId,
                                           item_index = movieId,
                                           rating = rating))

# Create reco model object
r_final <- recosystem::Reco()

# Select optimal tuning parameters
options_final <- r_final$tune(edx_reco, opts = list(dim = seq(10,40,10),
                                                    lrate = c(0.05, 0.1, 0.2),
                                                    nthread = 4,
                                                    costp_l2 = c(0.01, 0.1),
                                                    costq_l2 = c(0.01, 0.1),
                                                    niter = 10,
                                                    nfold = 10))

# Train final reco model
r_final$train(edx_reco, opts = c(options_final$min,
                                 nthread = 4,
                                 niter = 20,
                                 verbose = FALSE))

# Predict validation set using model
predict_final_reco <-  r_final$predict(valid_reco, out_memory())

# Calculate RMSE
final_rmse <- RMSE(predict_final_reco, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Matrix factorization on Validation Set",  
                                     RMSE = final_rmse ))
rmse_results %>% knitr::kable() %>% 
  kable_styling(latex_options=c("striped")) %>% 
  column_spec(1:2, color = "black") %>% 
  row_spec(0, color = "white", background = "#5a5cd6")










