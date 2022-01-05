#Load dataset
loan_data <- read.csv('~/Documents/Projects/Data Analyst/R/Loan_Prediction/LoanDataset.csv')

#################
# Install packages
# Creating Models
install.packages("caret", dependencies = TRUE)
library("caret")
#################
#Data Cleaning
loan <- na.omit(loan_data) # remove missing data

sum(is.na(loan)) # checking if missing data

# Data Value Conversion: Change values Y = 1, N = 0
loan$Loan_Status[loan$Loan_Status == "Y"] <- 1
loan$Loan_Status[loan$Loan_Status == "N"] <- 0

# remove Columns
loan = subset(loan, select = -c(Loan_ID))

##############
# Setting up Training and Testing Models: Purpose is to save the state of the 
# random function. Make sure to get the same results for randomization. 
set.seed(100) # set the random seed number (reproducible model)

# Split the Dataset: p = percentage of data that goes into training
TrainingIndex <- createDataPartition(loan$Loan_Status, p=0.8, list=FALSE )
TrainingSet <- loan[TrainingIndex,]
TestingSet <- loan[-TrainingIndex,] #remember to include the '-', p = .20 

# check datasets - first five
head(TrainingSet, 5)
head(TestingSet, 5)

# use View(loan)
#################
#Setting up Models: current model type - SVM Poly 
### Test out Rule-Based 

#Training model
Model <- train(as.factor(Loan_Status) ~ ., data=TrainingSet,
               method = "svmPoly",
               #na.action = na.omit(), #already removed missing data
               preProcess = c("scale", "center"), 
               trControl = trainControl(method="none"), 
               tuneGrid = data.frame(degree=1, scale=1, C=1)
               )

#Training model
Model.cv <- train(as.factor(Loan_Status) ~., data=TrainingSet,
               method = "svmPoly",
               preProcess = c("scale", "center"), 
               trControl = trainControl(method="cv", number=10), 
               tuneGrid = data.frame(degree=1, scale=1, C=1)
               )

#FIXME: Need to continue to clean Data

# Execute Prediction
Model.training <- predict(Model, TrainingSet)
Model.testing <- predict(Model, TestingSet)
Model.cv <- predict(Model.cv, TrainingSet)

# View Performance
Model.training.confusion <- confusionMatrix(Model.training, 
                                            as.factor(TrainingSet$Loan_Status))
Model.testing.confusion <- confusionMatrix(Model.testing, 
                                            as.factor(TestingSet$Loan_Status))
Model.cv.confusion <- confusionMatrix(Model.cv, 
                                            as.factor(TrainingSet$Loan_Status))

#Print Results
print(Model.training.confusion)
print(Model.testing.confusion)
print(Model.cv.confusion) 

#FIXME
Importance <- varImp(Model)
plot(Importance)
