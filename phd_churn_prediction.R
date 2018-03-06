# set working directory
setwd("E:/INSOFE/phd/Churn_prediction")

load("E:/INSOFE/phd/Churn_prediction/churn.RData")

#libraries

library(reshape2)
library(DMwR)
library(caret)
library(MLmetrics)
library(randomForest)
library(xgboost)
library(C50)
library(rpart)
library(ROCR)
library(ggplot2)

#reading all the train and test files

train <- read.csv("E:/INSOFE/phd/Churn_prediction/TrainData/Train.csv")
train_accinfo <- read.csv("E:/INSOFE/phd/Churn_prediction/TrainData/Train_AccountInfo.csv")
train_demograph <- read.csv("E:/INSOFE/phd/Churn_prediction/TrainData/Train_Demographics.csv") 
train_services <- read.csv("E:/INSOFE/phd/Churn_prediction/TrainData/Train_ServicesOptedFor.csv")

test <- read.csv("E:/INSOFE/phd/Churn_prediction/TestData/Test.csv")
test_accinfo <- read.csv("E:/INSOFE/phd/Churn_prediction/TestData/Test_AccountInfo.csv")
test_demograph <- read.csv("E:/INSOFE/phd/Churn_prediction/TestData/Test_Demographics.csv")
test_services <- read.csv("E:/INSOFE/phd/Churn_prediction/TestData/Test_ServicesOptedFor.csv")


dim(train)
dim(train_accinfo)
dim(train_demograph)
dim(train_services)

dim(test)
dim(test_accinfo)
dim(test_demograph)
dim(test_services)


names(train)
names(train_accinfo)
names(train_demograph)
names(train_services)

names(test)
names(test_accinfo)
names(test_demograph)
names(test_services)

dcast_data <- dcast(train_services,CustomerID ~ TypeOfService)
dcast_test_data <- dcast(test_services,CustomerID ~ TypeOfService)


View(dcast_data)
#merging the data sets based on the common attributes

merge1 <- merge(train,train_accinfo,by.x = "CustomerID",by.y = "CustomerID")
merge2 <- merge(dcast_data,train_demograph,by.x = "CustomerID",by.y = "HouseholdID")
merge3 <- merge(merge1,merge2,by.x = "CustomerID",by.y = "CustomerID")

test_merge1 <- merge(test,test_accinfo,by.x = "CustomerID",by.y = "CustomerID")
test_merge2 <- merge(dcast_test_data,test_demograph,by.x = "CustomerID",by.y = "HouseholdID")
test_merge3 <- merge(test_merge1,test_merge2,by.x = "CustomerID",by.y = "CustomerID")

dim(merge3)

#write.csv(merge3,"churn_data.csv")
#write.csv(test_merge3,"final_test.csv")

churn_data<- merge3
final_test <- test_merge3

dim(churn_data)
dim(final_test)

#churn_data <- read.csv("E:/INSOFE/phd/Churn_prediction/churn_data.csv")

dim(churn_data)

names(churn_data)

str(churn_data)

table(churn_data$DOC)
levels(churn_data$ContractType)
levels(churn_data$PaymentMethod)
levels(churn_data$Country)
levels(churn_data$State)
levels(churn_data$Education)
levels(churn_data$Gender)
levels(churn_data$Churn)

levels(final_test$ContractType)
levels(final_test$PaymentMethod)
levels(final_test$Country)
levels(final_test$State)
levels(final_test$Education)
levels(final_test$Gender)

levels(final_test$OnlineBackup)
levels(churn_data_imputed$OnlineBackup)

unique(churn_data$Retired)
unique(churn_data$HasPartner)
unique(churn_data$HasDependents)



#converting numeric into factor 

churn_data$Retired <- as.factor(churn_data$Retired)
churn_data$HasPartner <- as.factor(churn_data$HasPartner)
churn_data$HasDependents <- as.factor(churn_data$HasDependents)

final_test$Retired <- as.factor(final_test$Retired)
final_test$HasPartner <- as.factor(final_test$HasPartner)
final_test$HasDependents <- as.factor(final_test$HasDependents)

char_vars <- churn_data[sapply(churn_data,is.character)]
char_names <- names(char_vars)

churn_data$DeviceProtection <- as.factor(churn_data$DeviceProtection)
churn_data$HasPhoneService <- as.factor(churn_data$HasPhoneService)
churn_data$InternetServiceCategory <- as.factor(churn_data$InternetServiceCategory)
churn_data$MultipleLines <- as.factor(churn_data$MultipleLines)
churn_data$OnlineBackup <- as.factor(churn_data$OnlineBackup)
churn_data$OnlineSecurity <- as.factor(churn_data$OnlineSecurity)
churn_data$StreamingMovies <- as.factor(churn_data$StreamingMovies)
churn_data$StreamingTelevision <- as.factor(churn_data$StreamingTelevision)
churn_data$TechnicalSupport <- as.factor(churn_data$TechnicalSupport)


final_test$DeviceProtection <- as.factor(final_test$DeviceProtection)
final_test$HasPhoneService <- as.factor(final_test$HasPhoneService)
final_test$InternetServiceCategory <- as.factor(final_test$InternetServiceCategory)
final_test$MultipleLines <- as.factor(final_test$MultipleLines)
final_test$OnlineBackup <- as.factor(final_test$OnlineBackup)
final_test$OnlineSecurity <- as.factor(final_test$OnlineSecurity)
final_test$StreamingMovies <- as.factor(final_test$StreamingMovies)
final_test$StreamingTelevision <- as.factor(final_test$StreamingTelevision)
final_test$TechnicalSupport <- as.factor(final_test$TechnicalSupport)

#converting factor in numeric

churn_data$TotalCharges <- as.numeric(churn_data$TotalCharges)

final_test$TotalCharges <- as.numeric(final_test$TotalCharges)

# Here if you see the columns "country" and "state", there are two levels in each,
#where the first level is India and Maharashtra respectively and the other is missing
#values denoted by '?'
#Hence dropping these two colomns 

churn_data$Country <- NULL
churn_data$State <- NULL
churn_data$DOC<- NULL
churn_data$CustomerID <- NULL

final_test$Country <- NULL
final_test$State <- NULL
final_test$DOC<- NULL


# now checking the missing values
sum(is.na(churn_data))
na_count <-sapply(churn_data, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count$features <- names(churn_data)
na_count$sl <- c(1:nrow(na_count))
colnames(na_count) <- c("count","features","sl")
View(na_count)




str(churn_data$ContractType)


#education has missing values with "blanks"
#gender has missing values with "blanks".

#changing missing levels in the training data as NA

levels(churn_data$Education)[levels(churn_data$Education)== ""]<- NA
levels(churn_data$Gender)[levels(churn_data$Gender)==""]<- NA

levels(churn_data$Education)
levels(churn_data$Gender)

sum(is.na(churn_data))

#Now impute these missing values using mode

churn_data_imputed <- centralImputation(churn_data)
sum(is.na(churn_data_imputed))

levels(churn_data_imputed$Education)
levels(churn_data_imputed$Gender)



# now checking the missing values on test

sum(is.na(final_test))
na_count_test <-sapply(final_test, function(y) sum(length(which(is.na(y)))))
na_count_test <- data.frame(na_count_test)
na_count_test$features <- names(final_test)
na_count_test$sl <- c(1:nrow(na_count_test))
colnames(na_count_test) <- c("count","features","sl")
View(na_count_test)

levels(final_test$Education)[levels(final_test$Education)== ""]<- NA
levels(final_test$Gender)[levels(final_test$Gender)==""]<- NA

str(churn_data_imputed)

#imputation:

final_test_imputed <- centralImputation(final_test)

sum(is.na(final_test_imputed))

churn_data_imputed$CustomerID<- NULL
#churn_data_imputed$DOE <- as.numeric(churn_data_imputed$DOE)
churn_data_imputed$DOE <- NULL

final_test_imputed$CustomerID<- NULL
final_test_imputed$DOE <- NULL


table(churn_data_imputed$PaymentMethod)

table(final_test_imputed$PaymentMethod)

churn_data_final <- churn_data_imputed

levels(churn_data_final$PaymentMethod)[levels(churn_data_final$PaymentMethod)%in%c("Bank transfer (automatic)","Credit card (automatic)")] <- "Bank transfer (automatic)"

levels(churn_data_final$PaymentMethod)
table(churn_data_final$PaymentMethod)


#relevelling some factors

char_names <- names(char_vars)


#Exploratory Data Analysis

names(churn_data)

#class Distribution
table(churn_data$Churn)
plot(churn_data$Churn,col = "blue2",main = "class_distribution",xlab ="churn",ylab="count" )

names(churn_data)

#Base charge

num_data <- churn_data[,c("BaseCharges","TotalCharges")]

plot(churn_data$BaseCharges,churn_data$TotalCharges)

library(corrplot)

corrplot(cor(num_data))

#boxplot
boxplot(num_data)

#ElectronicBilling

ggplot(data = churn_data,aes(x=ElectronicBilling))+geom_bar(fill = "blue2")+labs(title = "frequency of Electroninc Billing")

#contract type

ggplot(data = churn_data_final,aes(x =ContractType))+geom_bar(fill ="blue2")+labs(title = "frequency of Contracttype")

#churn Vs Totalcharges

ggplot(data = churn_data_final,aes(x=(TotalCharges),fill = Churn))+geom_histogram(position = "fill")+labs(title = "churn vs totalcharges")

#churn and education

ggplot(data = churn_data_final,aes(x=Education,fill = Churn))+geom_histogram(stat = "count")+labs(title = "Education Vs Churn")

names(churn_data_final)

#InternetServiceCetegory
InternetServiceCategory

ggplot(data = churn_data_final,aes(x = ))


#model_building

#Train and validation split

set.seed(1000)
train.indices <- createDataPartition(churn_data_final$Churn,p=.80,list = F)
trainingData <- churn_data_final[train.indices,]
validData <- churn_data_final[-train.indices,]


logistic_model <- glm(Churn ~., data = trainingData,family = binomial)
names(trainingData)
train_pred <- predict(logistic_model,trainingData[,-1],type = 'response')
#ROCR curve for threshold

pred <- prediction(train_pred, trainingData$Churn)

perf <- performance(pred, measure="tpr", x.measure="fpr")
perf

#plotting the ROC curves

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")

# Access the area under curve(auc) score of roc curve from the performance object

auc <- perf_auc@y.values[[1]]
print(auc)

final_train_pred  <- ifelse(train_pred > 0.25, "Yes", "No")
confusionMatrix(final_train_pred,trainingData$Churn,positive = 'Yes')

Recall(trainingData$Churn,final_train_pred,positive = 'Yes')
table(final_train_pred)
table(trainingData$Churn)
valid_pred <- predict(logistic_model,validData[,-1],type = 'response')
final_valid_pred  <- ifelse(valid_pred > 0.25, "Yes", "No")
confusionMatrix(final_valid_pred,validData$Churn,positive = 'Yes')

table(final_valid_pred)
table(validData$Churn)

table(valid_pred)
test_pred <- predict(logistic_model,final_test_imputed,type = "response")


Churn <- ifelse(test_pred > 0.25, "Yes", "No")
table(Churn)

str(churn_data_imputed$Education)
str(final_test_imputed$Education)


#logistic with important variables

logistic_model <- glm(Churn ~., data = trainingData_imp,family = binomial)
names(trainingData)
train_pred <- predict(logistic_model,trainingData_imp[,-1],type = 'response')
#ROCR curve for threshold

pred <- prediction(train_pred, trainingData_imp$Churn)

perf <- performance(pred, measure="tpr", x.measure="fpr")
perf

#plotting the ROC curves

plot(perf, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc <- performance(pred, measure="auc")

# Access the area under curve(auc) score of roc curve from the performance object

auc <- perf_auc@y.values[[1]]
print(auc)

final_train_pred  <- ifelse(train_pred > 0.2, "Yes", "No")
confusionMatrix(final_train_pred,trainingData_imp$Churn,positive = 'Yes')

Recall(trainingData_imp$Churn,final_train_pred,positive = 'Yes')

valid_pred_imp <- predict(logistic_model,validData_imp[,-1],type = 'response')

final_valid_pred <- ifelse(valid_pred_imp > 0.2,"Yes","No")
confusionMatrix(final_valid_pred,validData_imp$Churn,positive = 'Yes')

table(final_valid_pred)
table(validData_imp$Churn)


#Random Forest

model_RF <- randomForest(Churn~.,data = trainingData,ntree = 1000)

train_pred <- predict(model_RF,trainingData[,-1])

confusionMatrix(train_pred,trainingData$Churn,positive = 'Yes')

varImpPlot(model_RF)
imp_var <-varImp(model_RF)


table(train_pred)
table(churn_data_final$Churn)

valid_pred <- predict(model_RF,validData[,-1])
confusionMatrix(valid_pred,validData$Churn,positive = 'Yes')

table(valid_pred)
table(validData$Churn)

Churn<- predict(model_RF,final_test_imputed)
table(churn)


#building with imp variables

imp_churn_data <- subset(churn_data_final,
                         select = c("Churn","BaseCharges","TotalCharges",
                        "ContractType","Education","PaymentMethod","OnlineSecurity",
                        "TechnicalSupport","InternetServiceCategory"))


set.seed(1000)
train.indices_imp <- createDataPartition(imp_churn_data$Churn,p=.80,list = F)
trainingData_imp <- imp_churn_data[train.indices_imp,]
validData_imp <- imp_churn_data[-train.indices,]

model_RF_imp <- randomForest(Churn~.,data = trainingData_imp,ntree = 100)

varImpPlot(model_RF_imp)

pred_train_imp <- predict(model_RF_imp,trainingData_imp[,-1])
confusionMatrix(pred_train_imp,trainingData_imp$Churn,positive = 'Yes')

pred_valid_imp <- predict(model_RF_imp,validData_imp[,-1])
table(pred_valid_imp)
table(validData_imp$Churn)
confusionMatrix(pred_valid_imp,validData_imp$Churn,positive = 'Yes')


names(churn_data_final)

#train and ctrl

#xgb

#create the training control parameters

xgb.ctrl <- trainControl(method = "repeatedcv", 
                         repeats = 5,number = 3,
                         search = 'random',
                         allowParallel = T)


xgb.tune <-train(Churn~., 
                 data = trainingData, 
                 method="xgbTree", 
                 trControl=xgb.ctrl,
                 # tuneGrid=xgb.grid, 
                 tuneLength=30, 
                 verbose=T, 
                 metric="Accuracy", 
                 nthread=3)
xgb.tune
View(xgb.tune$results)

train_pred <- predict(xgb.tune,trainingData[,-1])
confusionMatrix(train_pred,trainingData$Churn,positive = 'Yes')

valid_pred <- predict(xgb.tune,validData[,-1])
confusionMatrix(valid_pred,validData$Churn,positive = 'yes')

Churn <- predict(xgb.tune,final_test_imputed)
table(Churn)


#c50 and rpart

model_c50 <- C5.0(x = trainingData[,-1],y =trainingData[,1],rules = T)

summary(model_c50)
string <- model_c50$rules

cat(string)
pred_train <- predict(model_c50,trainingData_imp[,-1])
#pred_train <- as.data.frame(pred_train)
confusionMatrix(pred_train,trainingData_imp$Churn,positive = "Yes")

Churn <-predict(model_c50,final_test_imputed)
table(Churn)

#rpart 

names(trainingData_imp)

model_rpart <- rpart(Churn~.,data = trainingData)

pred_train<- predict(model_rpart,trainingData_imp[,-1],type = "class")
confusionMatrix(pred_train,trainingData$Churn)

plotcp(model_rpart)
printcp(model_rpart)

pruned_model <- prune(model_rpart,cp=0.010000 )
pred_train_prune <- predict(pruned_model,final_test_imputed)
plotcp(pruned_model)
printcp(pruned_model)

table(Churn)


CustomerID<- (test_merge3$CustomerID)

CustomerID <- as.data.frame(CustomerID)
Churn <- as.data.frame(Churn)
predictions <- cbind(CustomerID,Churn)

write.csv(predictions,"predictions.csv",row.names = F)

save.image("E:/INSOFE/phd/Churn_prediction/churn.RData")



##### Debugging 

dim(final_test_imputed)
dim(churn_data_imputed)
str(churn_data_imputed)
str(final_test_imputed)
