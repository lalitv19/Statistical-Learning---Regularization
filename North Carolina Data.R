#Multiple Linear regression on North Carolina data

#Libraries required
library(caret)
library(glmnet)
library(GGally)
library(coefplot)
library(psych)
library(dplyr)

#load the data fom online sv link
crime <- read.csv("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Crime.csv")
head(crime)

summary(crime)

str(crime)
#converted the factor variables into numeric for correlation 
crime$region <-as.numeric(crime$region)
crime$smsa <- as.numeric(crime$smsa)

#Check for missnig values
sum(is.na(crime))

multi.hist(crime)

#Correlation betwen predictors
ggcorr(crime, nbreaks = 4, label = TRUE)

#Data partition
set.seed(1234)
index <- sample(2, nrow(crime), 
                replace = T, 
                prob = c(0.70, 0.30))

traindata <- crime[index == 1, ]
testdata <- crime[index == 2, ]

#Custom Control Parameters to apply 10-Fold cross validation
custom <- trainControl(method = "cv",
                       number = 10)

#Multiple linear regression
set.seed(1234)
lm <- train(crmrte~.,
            traindata,
            method = 'lm',
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
lm.RMSE  <- RMSE(lm.predict, testdata$crmrte)
lm.RMSE
lm.R2    <- R2(lm.predict, testdata$crmrte)
lm.R2

###############################################
#Ridge Regularization
###############################################
set.seed(1234)
ridge <- train(crmrte ~ .,
               traindata,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 0.5, length=20)),
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

# RMSE and R2 for Ridge
ridge.RMSE  <- RMSE(ridge.predict, testdata$crmrte)
ridge.RMSE
ridge.R2    <- R2(ridge.predict, testdata$crmrte)
ridge.R2

####################################
#         Lasso                    #
####################################

# predictor variables 
x <- model.matrix(crmrte~., traindata)[,-1]

#outcome variable
y <- traindata$crmrte

set.seed(1234)
cv <- cv.glmnet(x, y, alpha = 1)

# Display the best lambda value
cv$lambda.min

# Fit the final model on the training data
lasso <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)

# Dsiplay regression coefficients
coef(lasso)

# Make predictions on the test data
x.test <- model.matrix(crmrte ~., testdata)[,-1]
lasso.predict <- lasso %>% predict(x.test) %>% as.vector()

# Model performance metrics
data.frame(
  RMSE = RMSE(lasso.predict, testdata$crmrte),
  Rsquare = R2(lasso.predict, testdata$crmrte)
)
###############################
# Elastic Net
###############################

set.seed(1234)
en <- train(
  crmrte ~., data = traindata, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10)

#find the best lmbda value
en$bestTune

coef(en$finalModel, en$bestTune$lambda)
x.test.net <- model.matrix(crmrte ~., testdata)[,-1]
en.predict <- en %>% predict(x.test.net)

data.frame(
  en.RMSE = RMSE(en.predict, testdata$crmrte),
  en.R2 = R2(en.predict, testdata$crmrte)
)