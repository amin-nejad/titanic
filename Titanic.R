#####################################################
# KAGGLE-"Titanic: Machine Learning from Disaster"
#####################################################

# The aim of this exercise is to predict whether certain passengers on the Titanic survived using
# Machine Learning, given the results from another set of passengers.

library(ggplot2)
library(ggcorrplot)
library(ggthemes)
library(scales)
library(plyr)
library(dplyr)
library(stats)
library(mice)
library(randomForest)
library(pastecs)

train <- read.csv('./train.csv', stringsAsFactors = F,header = TRUE)
test <- read.csv('./test.csv', stringsAsFactors = F,header=TRUE)
full  <- bind_rows(train, test) # Bind training & test data to help us extract the most value before prediction

#####################################################
#               USEFUL INFO. EXTRACTION
#####################################################

#############     TITLE MANIPULATION      ###########

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name) # Grab title from passenger names
table(full$Pclass, full$Title) # View all titles by class

# Reassign mlle, ms and mme accordingly
full$Title[full$Title=='Mlle']<-'Miss' 
full$Title[full$Title=='Ms']<-'Miss'
full$Title[full$Title=='Mme']<-'Mrs'

# Create a 'Rare Title' value to summarise all the "Upper Class" titles as there are too many different ones
# with not enough of each to be able to extract any meaninful value
rare_title <- unique(full$Title[!(full$Title %in% c("Mr","Master","Miss","Mrs"))])
full$Title[full$Title %in% rare_title]  <- 'Rare Title'
table(full$Pclass, full$Title) # View all titles by class again to show updated values

# Create a 'Rare Title' variable entirely - we will actually use this instead of the Title variable to avoid
# overfitting as the Title variable already contains information about the Sex and Age
full$Rare_Title[full$Title=="Rare Title"]<-"Y"
full$Rare_Title[full$Title!="Rare Title"]<-"N"

############    SURNAME AND FAMILY EXTRACTION   #############

full$Surname <- sapply(full$Name, function(x) strsplit(x, split = '[,.]')[[1]][1]) # Extract surname
nlevels(factor(full$Surname)) # No. of unique surnames
full$Fsize<-full$SibSp+full$Parch+1 #Create family size variable (Sibling/Spouse + Parents/Children + self)
# However Fsize variable does not account for passengers having both a sibling and a spouse meaning Fsize
# will not be a constant number within that family. Attempt to rectify below.
family<-as.data.frame(aggregate(Fsize~Pclass+Fare+Embarked+Surname, data=full, FUN=max))
colnames(family)[colnames(family)=="Fsize"] <- "Fsize_a"
full<-join(full,family,type="left",match="first")
full$Family <- paste(full$Surname, full$Fsize_a, sep='_')# Create unique family identifier
full$Fsize_a[is.na(full$Fsize_a)]<-full$Fsize[is.na(full$Fsize_a)]
#However doesn't account for those who haven't taken their spouse's surname or for co-incidences

# Looking at family size as a predictor; shows large families are not good for survival
ggplot(full[1:891,], aes(x = Fsize_a, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()

# Discretise family size to use as a predictor
full$FsizeD[full$Fsize_a == 1] <- 'Singleton'
full$FsizeD[full$Fsize_a < 5 & full$Fsize_a > 1] <- 'Small'
full$FsizeD[full$Fsize_a > 4] <- 'Large'

#############     EXTRACT  DECK        #############

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1])) # Extract Deck from Cabin
table(full$Deck,full$Pclass)
# As you can see from the table, the vast majority of Cabin values are empty with the populated values
# being largely in the First Class passenger category. This won't be very useful to us so we will leave it.

#####################################################
#                 VALUE IMPUTATION
#####################################################

#############       EMBARKMENT       #############
full$PassengerId[is.na(full$Embarked)|full$Embarked==""]
# Passengers 62 and 830 are missing Embarkment
embark_fare <- full %>% filter(PassengerId != 62 & PassengerId != 830)
#However, we can predict their Embarkment point given their ticket class and ticket price

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()
# Is is quite aparent that C is the most appropriate embarkment option given their 1st class tickets
# which had a price of 80. We will substitute this value in.
full$Embarked[c(62, 830)] <- 'C'

#############       FARE VALUE       #############

full$PassengerId[is.na(full$Fare)|full$Fare==""]
#Passenger on row 1044 has an NA Fare value
full[1044, ]
# He is a 3rd class passenger who departed from Southampton. Let's plot the fares for similar passengers
x<-ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], aes(x = Fare))
x<-x+geom_density(fill = '#99d6ff', alpha=0.4)
x<-x+geom_vline(aes(xintercept=median(Fare, na.rm=T)),colour='red', linetype='dashed', lwd=1)
x<-x+scale_x_continuous(labels=dollar_format())
x<-x+theme_economist_white()
x
# Assigning fare value for passenger 1004 as the median fare value for passengers of his type (~8)
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)

#############      AGE PREDICTION      ############

nlevels(factor(full$PassengerId[is.na(full$Age)|full$Age==""]))
# Age will undoubtedly be an important factor in predicting survival, but since there are 263 missing
# values, we will have to come up with another way to impute these values.
factor_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','Surname','Family','FsizeD')
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

set.seed(129) # Set a random seed
# Perform Random Forest imputation using the mice package, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','FSize','Deck','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)

# Comparing the age distribution using histograms pre and post imputation
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: RF Output', 
     col='lightgreen', ylim=c(0,0.04))
# We can see the distribution hasn't changed. So, we will proceed with updating the missing age values.
full2<-full
full2$Age <- mice_output$Age

# Looking closer at age values for Masters, there are 2 people with ages>18 which should not be the case.
nlevels(factor(full2$PassengerId[full2$Age>18&full2$Title=="Master"]))
# We will just replace these with the average for the rest of the Masters
full2$Age[full2$Age>18&full2$Title=="Master"]<-mean(full2$Age[full2$Title=="Master"],na.rm = TRUE)

###############       CREATE CHILD AND MOTHER VARIABLES   ##############

full2$Child[full2$Age < 18] <- 'Child'
full2$Child[full2$Age >= 18] <- 'Adult'
full2$Mother <- 'Not Mother'
full2$Mother[full2$Sex == 'female' & full2$Parch > 0 & full2$Age > 18 & full2$Title != 'Miss'] <- 'Mother'
full2$Child  <- factor(full2$Child)
full2$Mother <- factor(full2$Mother)

####################################################################
#                  BUILD SURVIVAL PREDICTOR MODEL
####################################################################

train <- full2[!(is.na(full2$Survived)),]
test <- full2[is.na(full2$Survived),]

set.seed(754)# Set a random seed
# Build the random forest model for survival (not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + FsizeD + Title + Child + Mother,data = train)

################        MODEL ERROR       ####################

# Take a look at model error by survival type
par(mfrow=c(1,1))
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
# Much more successful at predicting death than survival

##############        VARIABLE IMPORTANCE       ################

# Let's take a look at relative variable importance by plotting the mean decrease in Gini across all trees
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  labs(x = 'Variables') +
  coord_flip()

# Title, sex and Pclass are the most important variables in the model

############     CORRELATION MATRIX    ############# (just out of curiosity)

corr_data<-train
corr_data$Embarked <- revalue(train$Embarked, c("S" = 1, "Q" = 2, "C" = 3))
corr_data$Sex <- revalue(train$Sex, c("male" = 1, "female" = 2))
corr_data$Title <- revalue(train$Title, c("Master" = 1, "Mr" = 2,"Miss" = 3, "Mrs" = 4,"Rare Title" = 5))
corr_data$FsizeD <- revalue(train$FsizeD, c("Singleton" = 1, "Small" = 2, "Large" = 3))
corr_data$Child <- revalue(train$Child, c("Child" = 1, "Adult" = 2))
corr_data$FsizeD <- as.numeric(train$FsizeD)
corr_data$Child <- as.numeric(train$Child)
corr_data$Sex <- as.numeric(train$Sex)
corr_data$Embarked <- as.numeric(train$Embarked)
corr_data$Title <- as.numeric(train$Title)
corr_data$Pclass <- as.numeric(train$Pclass)
corr_data$Survived <- as.numeric(train$Survived)

corr_data <-corr_data[,c("Survived", "Pclass", "Sex", "FsizeD", "Fare", "Embarked","Title","Child")]
matrix_corr_data <- cor(corr_data) # Create correlation matrix
ggcorrplot(matrix_corr_data) # Visualise correlation matrix

################      PREDICT     ##################

prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
# Write the solution to file
write.csv(solution, file = './rf_mod_Solution.csv', row.names = F)

# Done!