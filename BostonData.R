#Multiple Linear regression on boston data

#Libraries required
library(MASS)
library(caret)
library(glmnet)
library(GGally)
library(coefplot)
library(psych)

#load the data
data("Boston")
head(Boston)

summary(Boston)

str(Boston)

#Check for missnig values
sum(is.na(Boston))

multi.hist(Boston)

#Correlation betwen predictors
ggcorr(Boston, nbreaks = 4, label = TRUE)

#Data partition
set.seed(124)
index <- sample(2, nrow(Boston), 
                replace = T, 
                prob = c(0.70, 0.30))

traindata <- Boston[index == 1, ]
testdata <- Boston[index == 2, ]

#Custom Control Parameters to apply 10-Fold cross validation
custom <- trainControl(method = "cv",
                       number = 10)

#Multiple linear regression
set.seed(124)
lm <- train(crim~.,
            traindata,
            method = 'lm',
            preProcess = c("center","scale"), 
            trControl = custom)
  
#Results
lm$results
lm
lm.summary <- summary(lm)
lm.summary

#Coefficients of Linear model
lm.coef <- coef(lm$finalModel)
lm.coef
coefplot(lm)

# Prediction
lm.predict <- predict(lm, testdata)

# Multiple regression RMSE and R2 values
lm.RMSE  <- RMSE(lm.predict, testdata$crim)
lm.RMSE
lm.R2    <- R2(lm.predict, testdata$crim)
lm.R2

###############################################
#Ridge Regularization
set.seed(124)
ridge <- train(crim ~ .,
               traindata,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 4, length=20)),
               preProcess = c("center","scale"), 
               trControl = custom)

summary(ridge)
ridge

#PLot results
plot(ridge)

ridge$bestTune # best lambda value
ridge$results

plot(ridge$finalModel, xvar = "lambda", label = T)
plot(ridge$finalModel, xvar = "dev", label = T) #variation in data

plot(varImp(ridge, scale=F)) # to find which variable is most important

#Coefficients of Ridge model
ridge.coef <- coef(ridge$finalModel, s=ridge$bestTune$lambda)
ridge.coef

# Prediction
ridge.predict <- predict(ridge, testdata)

# Multiple regression RMSE
ridge.RMSE  <- RMSE(ridge.predict, testdata$crim)
ridge.RMSE
ridge.R2    <- R2(ridge.predict, testdata$crim)
ridge.R2


###########################
#         Lasso           #
###########################
set.seed(124)
lasso <- train(crim ~ .,
               traindata,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1,
                                       lambda = seq(0.0001, 1.5, 
                                                    length = 20)),
               preProcess = c("center","scale"), 
               trControl = custom)

#Check lasso model
lasso

#Plot results for lasso
plot(lasso)

lasso$bestTune

plot(lasso$finalModel, xvar = 'lambda', label = T)
plot(lasso$finalModel, xvar = 'dev', label = T)
plot(varImp(lasso, scale = F))

#variable coefficients form lasso model
lasso.coef <- coef(lasso$finalModel, s=lasso$bestTune$lambda)
lasso.coef

# Prediction
lasso.predict <- predict(lasso, testdata)

#RMSE and R2 values
lasso.RMSE  <- RMSE(lasso.predict, testdata$crim)
lasso.RMSE

lasso.R2    <- R2(lasso.predict, testdata$crim)
lasso.R2

# Elastic Net
set.seed(124)
en <- train(crim ~ .,
            traindata,
            method = 'glmnet',
            tuneGrid = expand.grid(alpha = seq(0,1, length = 10),
                                        lambda = seq(0.0001, 2, length = 20)),
            preProcess = c("center","scale"),
            trControl = custom)

#check the model
en

# Plot
plot(en) 

en$bestTune # best lambda value

plot(en$finalModel, xvar = 'lambda', label = T) 
plot(en$finalModel, xvar = 'dev', label = T)
plot(varImp(en, scale = F))

# Best model
en$bestTune
best <- en$finalModel

coef(best, s = en$bestTune$lambda)
en.coef <- coef(en$finalModel, s=en$bestTune$lambda)

# Prediction
en.predict <- predict(en, testdata)

# RMSE and R2 values for elastic net model
en.RMSE  <- RMSE(en.predict, testdata$crim)
en.RMSE
en.R2    <- R2(en.predict, testdata$crim)
en.R2

#################################
#      Compare all the Models   #
#################################
model.list <- list(LinearModel = lm, 
                   Ridge = ridge, 
                   Lasso = lasso, 
                   ElasticNet = en)

compare <- resamples(model.list)
summary(compare)
