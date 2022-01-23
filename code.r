# Case study: employee retention
library(dplyr)
library(rpart)
library(ggplot2)
library(scales)
data = read.csv("employee_retention_data.csv") # load data
str(data) #check the structure
# transform the variables into appropriate format
data$company_id = as.factor(data$company_id) # this is a categorical var
data$join_date = as.Date(data$join_date) #make it a date
data$quit_date = as.Date(data$quit_date) #make it a date
summary(data) # everything seems to make sense, some simple plots would help double check

# Aggregate information
# create a table with 3 columns: day, employee_headcount, company_id.
unique_dates = seq(as.Date("2011/01/24"), as.Date("2015/12/13"), by = "day") # create list of unique dates for the table
unique_companies = unique(data$company_id) #ceate list of unique companies
data_headcount = merge(unique_dates, unique_companies, by = NULL) #cross join to get all combinations of dates and companies. Will need it later.
colnames(data_headcount) = c("date", "company_id")
head(data_headcount)

# get for each day/company, how many people quit/got hired on that day
data_join = data %>% group_by(join_date, company_id) %>% summarise(join_count = length(join_date))
data_join

data_quit = data %>% group_by(quit_date, company_id) %>% summarise(quit_count = length(quit_date))
data_quit

#Now left outer join with data_headcount.
#NA means no people were hired/quit on that day.
data_headcount = merge (data_headcount, data_join, by.x = c("date", "company_id"), by.y = c("join_date", "company_id"), all.x = TRUE)
head(data_headcount)

data_headcount = merge (data_headcount, data_quit, by.x = c("date", "company_id"), by.y = c("quit_date", "company_id"), all.x = TRUE)
head(data_headcount)

#replace the NAs with 0
data_headcount$join_count[is.na(data_headcount$join_count)] = 0
data_headcount$quit_count[is.na(data_headcount$quit_count)] = 0
head(data_headcount)

#Now we need the sum by company_id. Data set is already ordered by date,
# so we can simply use dplyr to group by company_id and do cumsum
data_headcount = data_headcount %>% group_by(company_id) %>% mutate ( join_cumsum = cumsum(join_count), quit_cumsum = cumsum(quit_count))
data_headcount[1:20,]

# finally, for each date I just take join_count - quit_count and I am done
data_headcount$count = data_headcount$join_cumsum - data_headcount$quit_cumsum
data_headcount[1:20,]

data_headcount_table = data.frame(data_headcount[, c("date", "company_id","count")])
head(data_headcount_table)


# let's try to understand employee retention before feature engineering
# how many days was she employed? This should matter.
# people might get bored in the same place for too long
data$employment_length = as.numeric(data$quit_date - data$join_date)
head(data$employment_length)

# plot employee length in days
hist(data$employment_length, breaks = 100)
# Very interesting, there are peaks around each employee year anniversary


#In general, whenever you have a date, extract week of the year, and day of the week. They tend to give an idea of seasonlity and weekly trends.
#In this case, weekly trends probably don't matter. So let's just get week of the year
data$week_of_year = as.numeric(format(data$quit_date, "%U"))
head(data$week_of_year)


#plot week of the year
hist(data$week_of_year, breaks = length(unique(data$week_of_year)))
# We see peaks around the new year. Makes sense, companies have much more money to hire at the beginning of the year.

# Now, let’s see if we find the characteristics of the people who quit early. Looking at the histogram of employment_length, it looks like we could define early quitters as those people who quit within 1 yr or so. So, let’s create two classes of users: quit within 13 months or not (if they haven’t been in the current company for at least 13 months, we remove them).
#Create binary class
data = subset(data, data$join_date < as.Date("2015/12/13") - (365 + 31)) # only keep people who had time to age
data$early_quitter = as.factor(ifelse( is.na(data$quit_date) | as.numeric(data$quit_date - data$join_date) > 396, 0, 1))
head(data$early_quitter)

# Let’s now build a model. Here we can just care about: seniority, salary, dept and company. You can build any classification models. A simple decision tree is probably good enough.
tree = rpart(early_quitter ~ .,data[, c("company_id", "dept", "seniority", "early_quitter", "salary")], control = rpart.control(minbucket = 30, maxdepth = 3), parms = list(prior = c(0.5,0.5)))
# minbucket: the minimum number of observations in any terminal <leaf> node
# 
tree #we are not too interested in predictive power, we are mainly using the tree as a descriptive stat tool.

#Not very surprising! Salary is what matters the most. After all, it probably has within it information about the other variables too. That is, seniority, dept and company impact salary. So salary carries pretty much all the information available.

#looking at the terminal nodes, the way the tree split is: If salary between 62500 and 224500, the employee has higher probability of being an early quitter, otherwise she doesn't. That means that people who make a lot of money and little are not likely to quit.

#By plotting the proportion of early quitter by salary percentile, this becomes quite clear:
data$salary_percentile = cut(data$salary, breaks = quantile(data$salary, probs = seq(0, 1, 0.02)), include.lowest = TRUE, labels = 1:50)
data_proportion_by_percentile = data %>% group_by(salary_percentile) %>%summarize(proportion_early_quitters = length(early_quitter[early_quitter==1])/length(early_quitter))
qplot(salary_percentile, proportion_early_quitters, data=data_proportion_by_percentile, geom="line", group =1) + scale_x_discrete(breaks = seq(1,50, by=2))
########################################################

Adding one variable: 
Given how important is salary, I would definitely love to have as a variable the salary that a quitter was offered in the next job. Otherwise, things like: promotions or raises received during the employee tenure would be interesting.

Conclusions
1. The major findings are that employees quit at year anniversaries or at the beginning of the year. Both cases make sense. Even if you don’t like your current job, you often stay for 1 yr before quitting + you often get stocks after 1 yr so it makes sense to wait. Also, the beginning of the year is well known to be the best time to change job: companies are hiring more and you often want to stay until end of Dec to get the calendar year bonus.
2. Employees with low and high salaries are less likely to quit. Probably because employees with high salaries are happy there and employees with low salaries are not that marketable, so they have a hard time finding a new job.



