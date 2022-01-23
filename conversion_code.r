# Download the data "conversion.csv" and install the four packages below 
library(dplyr)
library(rpart)
library(ggplot2)
library(randomForest)
library(glmnet)
# Let’s read the dataset into R.
data = read.csv('conversion.csv')
# We should get something like this:
head(data)
# Let’s check the structure of the data:
str(data)


summary(data)
table(data$country)


sort(unique(data$age), decreasing=TRUE)


subset(data, age>79)


data = subset(data, age<80)


data_country = data %>% group_by(country) %>% summarise(conversion_rate = mean(converted))

ggplot(data=data_country, aes(x=country, y=conversion_rate)) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=country), vjust=-0.3, size=3.5)+
  theme_minimal()




data_pages = data %>% group_by(total_pages_visited) %>% summarise(conversion_rate = mean(converted))
ggplot(data=data_pages, aes(x=total_pages_visited, y=conversion_rate, group=1)) +
  geom_line()+
  geom_point()


# logistic model:
x = model.matrix(converted~., data)[,-1]
y = data$converted 
# cross validation 
# cvlogit = cv.glmnet(x, y, alpha = 0, family = "binomial") # alpha=0 for l2 penalty
model = glmnet(x, y, alpha = 0, family = "binomial", lambda = 0.01)#cvlogit$lambda.min)
coef(model)
# training accuracy
yhat =1 * (as.numeric(cbind(1,x)%*%coef(model))>0 )
print(paste('training accuracy: ', mean(yhat==y), sep=''))



## Next, We are going to use a random forest to predict conversion rate. 

# random forests:
# Firstly, "Converted" should really be a factor here as well as new_user. So let's change them:
data$converted = as.factor(data$converted) # let's make the class a factor
data$new_user = as.factor(data$new_user) #also this a factor
levels(data$country)[levels(data$country)=="Germany"]="DE" # Shorter name, easier to plot.



train_sample = sample(nrow(data), size = nrow(data)*0.66)
train_data = data[train_sample,]
test_data = data[-train_sample,]
rf = randomForest(y=train_data$converted, x = train_data[, -ncol(train_data)],
ytest = test_data$converted, xtest = test_data[, -ncol(test_data)], ntree = 100, mtry = 3, keep.forest = TRUE)
rf


 
# Let’s start checking variable importance:
varImpPlot(rf,type=2)


# Let’s rebuild the RF without Total pages visited. 

rf = randomForest(y=train_data$converted, x = train_data[, -c(5, ncol(train_data))], ytest = test_data$converted, xtest = test_data[, -c(5, ncol(train_data))], ntree = 100, mtry = 3, keep.forest = TRUE, classwt = c(0.7,0.3)) #classwt: Priors of the classes. Need not add up to one. Ignored for regression.
rf


varImpPlot(rf,type=2)


# Interesting! New user is the most important one. Source doesn’t seem to matter at all.
# Let’s check partial dependence plots for the 4 vars:


op <- par(mfrow=c(2, 2))
partialPlot(rf, train_data, country, 1)
partialPlot(rf, train_data, age, 1)
partialPlot(rf, train_data, new_user, 1)
partialPlot(rf, train_data, source, 1)
# partial dependence shows the marginal effect of one or two features on the predicted outcome of a machine learning model (J. H. Friedman). A partial dependence plot can show whether the relationship between the target and a feature is linear, monotonic or more complex. For example, when applied to a linear regression model, partial dependence plots always show a linear relationship.
# In partial dependence plots, we just care about the trend, not the actual y value. So this shows that:

# Let’s now build a simple decision tree and check the important segments. You can include more more or less variables:
head( data[, -c(5, ncol(data))])
tree = rpart(data$converted ~ ., data[, -c(5, ncol(data))], control = rpart.control(maxdepth = 3), parms = list(prior = c(0.7, 0.3)))
tree





