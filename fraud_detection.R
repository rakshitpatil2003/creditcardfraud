#import the dataset from the respective folder
credit_card <- read.csv("C:\\Users\\Asus\\Documents\\miniproject_data\\creditcardb.csv")

#Glance at the structure of the dataset
str(credit_card)

#convert class to factor variable
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

#get the summary of data
summary(credit_card)


#count the missing values
sum(is.na(credit_card))

#..................................................................................

#get the distribution of fraud and legit transactions
table(credit_card$Class)

#get the % of fraud and legit
prop.table(table(credit_card$Class))

#pie chart of credit card transaction
labels <- c("legit", "fraud")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)), 2))
labels <- paste(labels, "%")

pie(table(credit_card$Class), labels, col = c("orange", "red"),
    main = "Pie chart of credit card transactions")


#..........................................................................................
#No model predictions

predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0,1))

#install.packages ('caret')
library(caret)
confusionMatrix(data = predictions, reference = credit_card$Class)


#...........................................................................................

library(dplyr)                      

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)       

table(credit_card$Class)

library(ggplot2)

ggplot(data = credit_card, aes(x =V1, y=V2, col = Class)) +
  geom_point() +
  theme_bw()+
  scale_color_manual(values = c('dodgerblue2', 'red'))


#.................................................................................................

#creating training and test sets of fraud detection

#install.packages('caTools')
library(caTools)

set.seed(123)

data_sample = sample.split(credit_card$Class, SplitRatio=0.80)

train_data = subset(credit_card,data_sample==TRUE)

test_data = subset(credit_card , data_sample==FALSE)

dim(train_data)
dim(test_data)


#...................................................................................................
#Random over-sampling

table(train_data$Class)

n_legit<- 22750
new_frac_legit<- 0.50
new_n_total <- n_legit/new_frac_legit #=22750/0.50

#install.packages("ROSE")
library(ROSE)
oversampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = "over",
                                   N = new_n_total,
                                   seed = 2019)

oversampled_credit <- oversampling_result$data
table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x=V1, y=V2, col=Class))+
  geom_point(position=position_jitter(width=0.2))+
  theme_bw()+
  scale_color_manual(values=c('dodgerblue2','red'))

#...................................................................................
#Random undersampling

table(train_data$Class)

n_fraud<-35
new_frac_fraud <- 0.50
new_n_total<- n_fraud/new_frac_fraud #=35/0.50

#library(ROSE)
undersampling_result <- ovun.sample(Class ~ .,
                                    data=train_data,
                                    method="under",
                                    N=new_n_total,
                                    seed=2019)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

ggplot(data = undersampled_credit, aes(x=V1, y=V2, col=Class))+
  geom_point(position=position_jitter(width=0.2))+
  theme_bw()+
  scale_color_manual(values=c('dodgerblue2','red'))


#...................................................................................
#using ros and rus

n_new <- nrow(train_data) #=22785
fraction_fraud_new <- 0.5

sampling_result <- ovun.sample(Class ~ .,
                                    data=train_data,
                                    method="both",
                                    N = n_new,
                                    p = fraction_fraud_new,
                                    seed=2019)

sampled_credit <- sampling_result$data
table(sampled_credit$Class)

prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x=V1, y=V2, col=Class))+
  geom_point(position=position_jitter(width=0.2))+
  theme_bw()+
  scale_color_manual(values=c('dodgerblue2','red'))


#.................................................................................

#using smote to balance the dataset

#install.packages("smotefamily")
library(smotefamily)

table(train_data$Class)

#set the number of fraud and legistimate, and the desired % of legitimate cases

n0<-22750
n1<- 35
r0<- 0.6

#calculate the value for the dup size parameter of smote
ntimes <- ((1-r0)/r0)* (n0/n1)-1

smote_output=SMOTE(X=train_data[ , -c(1,31)],
                   target = train_data$Class,
                   K=5,
                   dup_size=ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

#class distribution for original dataset
ggplot(data = train_data, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  scale_color_manual(values=c('dodgerblue2','red'))


#class distribution for oversampled dataset
ggplot(data = credit_smote, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  scale_color_manual(values=c('dodgerblue2','red'))


#........................................................................................
#install.packages("rpart")
#install.packages("rpart.plot")

library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ . , credit_smote)

rpart.plot(CART_model, extra = 0, type =5, tweak =1.2)

#predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

#Build confusion matrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)

#......................................................................................................


predicted_val <- predict(CART_model, credit_card[,-1], type ='class')
confusionMatrix(predicted_val, credit_card$Class)

#......................................................................................................

#Decision tree without smote

CART_model <- rpart(Class ~ . , train_data[,-1])

rpart.plot(CART_model, extra =0, type = 5, tweak = 1.2)

#predict fraud cases
predicted_val<- predict(CART_model, test_data[,-1], type ='class')

library(caret)
confusionMatrix(predicted_val, test_data$Class)


#......................................................................................................

predicted_val <- predict(CART_model, credit_card[,-1], type ='class')
confusionMatrix(predicted_val, credit_card$Class)

#----------------------------------------------------------------------------------------------------


