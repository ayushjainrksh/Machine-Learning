dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


#Fitting the dataset in the Linear Regression model
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

#Fittng the dataset in the Polynomial Regression model
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)


library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth and bluff(Linear Regression)') +
  xlab('Level') +
  ylab('Salary')


library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             color = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            color = 'blue') +
  ggtitle('Truth and bluff(Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

y_pred = predict(lin_reg, data.frame(Level=6.5))

y_pred_new = predict(poly_reg, data.frame(Level=6.5,
                                          Level2=6.5^2,
                                          Level3=6.5^3,
                                          Level4=6.5^4))