# Data processing
library(dplyr)

MH_data <- read.table("MHpredict.csv", sep = ",", header = TRUE)
MH_data <- mutate_if(MH_data, is.character, as.factor)
MH_data <- mutate_if(MH_data, is.logical, as.factor)
MH_data <- mutate_if(MH_data, is.integer, as.numeric)


set.seed(2814412)
n <- nrow(MH_data)
shuffle.ind <- sample(n,1152)
MH_data.shuffle <- MH_data[shuffle.ind,]
train <- MH_data.shuffle[1:1000,]
test <- MH_data.shuffle[1001:n,]

summary(MH_data)


# linear regression
reg <- lm(dep_sev_fu ~ (.), data = train)
res <- resid(reg)
pred.lm <- round(predict(reg,test,type = "response"),0)

mean((test[,21]-pred.lm)^2)
par(mfrow = c(2,2))
plot(reg)



# Random Forest
library("randomForest")
library("caret")

grid <- expand.grid(mtry = 4:12)
set.seed(2814412)

fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 5)

rfFit <- caret::train(dep_sev_fu ~ . , data = train, tuneGrid = grid,
                      trControl = fitControl,
                      method = "rf", 
                      verbose = FALSE, metric = "RMSE")

print(rfFit, showSD = TRUE, digits = 3)
plot(rfFit)
rfFit$bestTune
set.seed(2814412)
rf.ens <- randomForest(dep_sev_fu ~ . , data = train,
                       importance = TRUE, mtry = 8)
preds <- round(predict(rf.ens, newdata =test, type = "response"),0)
mean((test$dep_sev_fu - preds)^2)
importance(rf.ens)
varImpPlot(rf.ens, cex = .6)
partialPlot(rf.ens, x.var = "IDS", pred.data = train)
partialPlot(rf.ens, x.var = "disType", pred.data = train)


# Stochastic gradient boosting


library("gbm")
library("caret")

grid <- expand.grid(shrinkage = c(.1, .01, .005, .001),
                    n.trees = c(10, 100, 1000),
                    interaction.depth = 1:4,
                    n.minobsinnode = c(10,20,30))


set.seed(2814412)

fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated ten times
  repeats = 5)

gbmFit <- caret::train(dep_sev_fu ~ . , data = train, 
                       method = "gbm",
                       tuneGrid = grid,
                       trControl = fitControl,
                       distribution = "gaussian", 
                       verbose = FALSE, metric = "RMSE")


print(gbmFit, showSD = TRUE, digits = 3)
plot(gbmFit)
gbmFit$bestTune
set.seed(2814412)
boost.best <- gbm(dep_sev_fu ~ . , data = train, n.trees = 1000,
                  shrinkage = .01, interaction.depth = 2,
                  distribution = "gaussian", n.minobsinnode = 20)
preds <- round(predict(boost.best, newdata =test, type = "response"),0)
mean((test$dep_sev_fu - preds)^2)

rslt.gbm <- summary(boost.best, cex.lab = .4, cex.axis = .7, cex.sub = .4, cex = .4)
rslt.gbm$var
plot(boost.best, i.var = "IDS")
plot(boost.best, i.var = "disType")


# Deep learning
find_optimal_mse = function(history){
  df.his1 = as.data.frame(history)
  condition = (df.his1$metric == "mse") & (df.his1$data == "validation")
  df.his1.mse = df.his1[condition,]
  df.his1.mse.clean = na.omit(df.his1.mse)
  cat("min mse in validation set is:", min(df.his1.mse.clean$value) )
  return(df.his1.mse.clean)
}


library(Rcpp)
library(keras)
library(dbarts)
train.full <- makeModelMatrixFromDataFrame(train[,-21])
train.x <-  train.full[1:900,]
train.y <- train[1:900,21]
valdi.x <- train.full[901:1000,]
valid.y <- train[901:1000,21]
test.x <-  makeModelMatrixFromDataFrame(test[,-21])
test.y <-  test[,21]
dim(valdi.x)


## 2 hidden layers- less neurons

set.seed(2814412)
library("tensorflow")
set_random_seed(2814412)

model.nn0 <- keras_model_sequential() %>% 
  layer_dense(units = 128, activation = "sigmoid", input_shape = 27) %>%
  layer_dense(units = 64, activation = "sigmoid") %>% 
  layer_dense(units = 1)

model.nn0 %>% compile(
  optimizer = "adam",
  loss = "mean_squared_error", 
  metrics = c("mse"))


history0 <- model.nn0 %>% fit(train.x, train.y, 
                              epochs = 100,
                              validation_data = list(valdi.x,valid.y),
                              callbacks = list(callback_early_stopping(patience = 10,restore_best_weights = TRUE)), 
                              verbose = 0)

find_optimal_mse(history0)

## 2 hidden layers- more neurons

set.seed(2814412)
library("tensorflow")
set_random_seed(2814412)

model.nn1 <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "sigmoid", input_shape = 27) %>%
  layer_dense(units = 128, activation = "sigmoid") %>% 
  layer_dense(units = 1)

model.nn1 %>% compile(
  optimizer = "adam",
  loss = "mean_squared_error", 
  metrics = c("mse"))


history1 <- model.nn1 %>% fit(train.x, train.y, 
                              epochs = 100,
                              validation_data = list(valdi.x,valid.y),
                              callbacks = list(callback_early_stopping(patience = 10,restore_best_weights = TRUE)), 
                              verbose = 0)
find_optimal_mse(history1)

## 3 hidden layers

set.seed(2814412)
library("tensorflow")
set_random_seed(2814412)

model.nn2 <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "sigmoid", input_shape = 27) %>%
  layer_dense(units = 128, activation = "sigmoid") %>% 
  layer_dense(units = 64, activation = "sigmoid") %>% 
  layer_dense(units = 1)

model.nn2 %>% compile(
  optimizer = "adam",
  loss = "mean_squared_error", 
  metrics = c("mse"))


history2 <- model.nn2 %>% fit(train.x, train.y, 
                              epochs = 100,
                              validation_data = list(valdi.x,valid.y),
                              callbacks = list(callback_early_stopping(patience = 10,restore_best_weights = TRUE)),
                              verbose = 0)
find_optimal_mse(history2)

## 4 hidden layers

set.seed(2814412)
library("tensorflow")
set_random_seed(2814412)

model.nn3 <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "sigmoid", input_shape = 27) %>%
  layer_dense(units = 128, activation = "sigmoid") %>% 
  layer_dense(units = 64, activation = "sigmoid") %>% 
  layer_dense(units = 32, activation = "sigmoid") %>% 
  layer_dense(units = 1)

model.nn3 %>% compile(
  optimizer = "adam",
  loss = "mean_squared_error", 
  metrics = c("mse"))


history3 <- model.nn3 %>% fit(train.x, train.y, 
                              epochs = 100,
                              validation_data = list(valdi.x,valid.y),
                              callbacks = list(callback_early_stopping(patience = 10,restore_best_weights = TRUE)),
                              verbose = 0)

find_optimal_mse(history3)

## Best NNs model
metrics_deep <- model.nn0 %>% predict(test.x)
mean((round(metrics_deep,0) - test.y)^2)




# new data and prediction based on three models
DS_data <- read.table("DSthymia.csv", sep = ",", header = TRUE)
test.add.DS <- rbind(test[,-21] , DS_data)
DS.index <- nrow(test.add.DS)
all(DS_data == test.add.DS[DS.index,])

gbm.pre.DS = round(predict(boost.best, newdata = test.add.DS, type = "response"),0)[DS.index]
gbm.pre.DS

rf.pre.DS = round(predict(rf.ens, newdata = test.add.DS, type = "response"),0)[DS.index]
rf.pre.DS

test.add.DS.matrix <- makeModelMatrixFromDataFrame(test.add.DS)
dl.pre <- model.nn1 %>% predict(test.add.DS.matrix)
dl.pre.DS <- dl.pre[DS.index]
round(dl.pre.DS)





# Derive 95 percent paire-wise interval
set.seed(2814412)
rep <- 1000
test.x <-  makeModelMatrixFromDataFrame(test[,-21]) # for NNs
test.y <-  test[,21] # for NNs


rf.sgbm <- numeric(rep)
rf.nns <- numeric(rep)
sgbm.nns <- numeric(rep)

for (i in 1:1000){
  
  ind.subtest <- sample(1:nrow(test), 100 ,replace = TRUE)
  subtest <- test[ind.subtest,]
  
  sgbm.preds <- round(predict(boost.best, newdata =subtest, type = "response"),0)
  sgbm.mse <- mean((subtest$dep_sev_fu - sgbm.preds)^2)
  
  
  
  r.fs.preds <- round(predict(rf.ens, newdata =subtest, type = "response"),0)
  r.fs.mse <-mean((subtest$dep_sev_fu - r.fs.preds)^2)
  
  
  NNs <- model.nn0 %>% predict(test.x[ind.subtest,])
  NNs.mse <- mean((round(NNs,0) - test.y[ind.subtest])^2)
  
  rf.sgbm[i] <- r.fs.mse - sgbm.mse
  rf.nns[i] <- r.fs.mse - NNs.mse 
  sgbm.nns[i] <- sgbm.mse - NNs.mse 
}

paste("The 95 CI for MSErf - MSEsgbm is: (", quantile(rf.sgbm,0.05),
      ", ", quantile(rf.sgbm,0.95),")")
paste("The 95 CI for MSErf - MSEnns is: (", quantile(rf.nns,0.05),
      ", ", quantile(rf.nns,0.95),")")
paste("The 95 CI for MSEsgbm - MSEsnns is: (", quantile(sgbm.nns,0.05),
      ", ", quantile(sgbm.nns,0.95),")")


