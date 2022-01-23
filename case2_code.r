library(dplyr)
library(randomForest)
library(ROCR)
# load data
data = read.csv("Fraud_Data.csv")
ip_addresses = read.csv("IpAddress_to_Country.csv")
#are there duplicates?
nrow(data) == length(unique(data$user_id))
#Let's add the country to the original data set by using the ip address
data_country = rep(NA, nrow(data))
for (i in 1: nrow(data))
{
	country = as.character(ip_addresses [data$ip_address[i] >= ip_addresses$lower_bound_ip_address & data$ip_address[i] <= ip_addresses$upper_bound_ip_address, "country"])
	if (length(country) == 1) data_country[i] = country  
}
data$country = data_country

# Date and time conversion
data[, "signup_time"] = as.POSIXct(data[, "signup_time"], tz="GMT")
data[, "purchase_time"] = as.POSIXct(data[, "purchase_time"], tz="GMT")
summary(as.factor(data$country))


# create the features
# time difference between purchase and signup
data$purchase_signup_diff = as.numeric(difftime(as.POSIXct(data$purchase_time, tz="GMT"), as.POSIXct(data$signup_time, tz="GMT"), unit="secs"))
head(data$purchase_signup_diff)
#check for each device id how many different users had it
data = data %>% group_by(device_id) %>% mutate (device_id_count = n())
head(data$device_id_count)
#check for each ip address how many different users had it
data = data.frame(data %>% group_by(ip_address) %>% mutate (ip_address_count = n()) )
head(data$ip_address_count)
#day of the week
data$signup_time_wd = format(data$signup_time, "%A")
data$purchase_time_wd = format(data$purchase_time, "%A" )
head(data$signup_time_wd)
#week of the yr
data$signup_time_wy = as.numeric(format(data$signup_time, "%U"))
data$purchase_time_wy = as.numeric(format(data$purchase_time, "%U" ))
head(data$signup_time_wy)

summary(data)
# Drop user_id, signup_time, purchase_time and device_id
data_rf = data[, -c(1:3, 5)]

#replace the NA in the country var
data_rf$country[is.na(data_rf$country)]="Not_found"
#just keep the top 50 country, everything else is "other"
data_rf$country = ifelse(data_rf$country %in% names(sort(table(data_rf$country), decreasing = TRUE ) )[51:length(unique(data_rf$country))], "Other", as.character(data_rf$country))
table(data_rf$country)
#make class a factor
data_rf$class = as.factor(data_rf$class)
#all characters become factors
data_rf[sapply(data_rf, is.character)] <- lapply(data_rf[sapply(data_rf, is.character)], as.factor)

# train/test partition
train_sample = sample(nrow(data_rf), size = nrow(data)*0.66)
train_data = data_rf[train_sample,]
test_data = data_rf[-train_sample,]


# Let's run a random forests and see which features have big impacts
rf = randomForest(y=train_data$class, x = train_data[, -7], ytest = test_data$class, xtest = test_data[, -7], ntree = 100, mtry = 3, keep.forest = TRUE)
rf

# randomforest internally uses 0.5 as the threshold. We check ROC to see if we want to change the threshold
# false positive and false negative: build the ROC and look for possible cut-off points
rf_results = data.frame (true_values = test_data$class, predictions = rf$test$votes[,2]) # we use majority voting for heuristic probabilities
pred = prediction(rf_results$predictions, rf_results$true_values)
perf = performance (pred, measure = 'tpr', x.measure = "fpr")
plot(perf) + abline(a=0, b=1, col = 'red') 

varImpPlot(rf,type=2)
par(mfrow=c(3,3))
partialPlot(rf, train_data, purchase_signup_diff, 1)
partialPlot(rf, train_data, purchase_time_wy, 1)
partialPlot(rf, train_data, device_id_count, 1)
partialPlot(rf, train_data, ip_address_count, 1)
partialPlot(rf, train_data, signup_time_wy, 1)
partialPlot(rf, train_data, ip_address, 1)
partialPlot(rf, train_data, purchase_value, 1)
partialPlot(rf, train_data, age, 1)
partialPlot(rf, train_data, sex, 1)


