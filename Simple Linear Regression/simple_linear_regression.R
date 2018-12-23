#Data Preprocessing

dataset = read.csv('Salary_Data.csv')

library(caTools)
set.seed(123)

split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split==TRUE)
test_set = subset(dataset, split==FALSE)

#Fitting Simple Linear Regression to Training set
regressor = lm(formula = Salary ~ YearsExperience, training_set)

#Predicting the Test Set results
y_pred = predict(regressor  , newdata = test_set)

#Visualising the Training set result
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes( x = training_set$YearsExperience, y = training_set$Salary),
                  color = 'red') +
  geom_line(aes( x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Years of Experience(Training set)') +
  xlab("Years of Experience") +
  ylab("Salary")

ggplot() +
  geom_point(aes( x = test_set$YearsExperience, y = test_set$Salary),
             color = 'red') +
  geom_line(aes( x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            color = 'blue') +
  ggtitle('Salary vs Years of Experience(Test set)') +
  xlab("Years of Experience") +
  ylab("Salary")



             