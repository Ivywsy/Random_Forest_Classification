library(tidyverse)
library(tidymodels)
library(vip)

###############
# LOADING AND CHECKING DATA
###############
phone <- read.csv("dataset.csv")%>%
  mutate_if(is.character, as.factor)%>%
  filter(Time.owned.current.phone < quantile(Time.owned.current.phone,0.999)) #Remove outlines, keep the 99.9% quantile (1 record removed)

dim(phone)

###############
# EXPLORING THE DATA
###############
#See the proportion of class 0.32 vs 0.67
phone %>%
  count(Gender)%>%
  mutate(prop = n/sum(n))


###############
# SPLIT DATA INTO TRAIN AND TEST SET
###############
set.seed(123)
splits <- initial_split(phone, strata = Gender, prop=0.8)
phone_train <- training(splits)
phone_test <- testing(splits)

#Cross-validation set
set.seed(123)
val_set <- validation_split(phone_train, strata = Gender, prop = 0.8)


###############
# SET UP THE 1ST RF MODEL
###############
#detect the number of cores in the computer
cores <- parallel::detectCores()

rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees=tune()) %>%  #use "tune()" to auto-tune the parameters
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

rf_recipe <- 
  recipe(Gender ~ ., data = phone_train) %>%
  step_normalize(all_numeric()) %>%
  step_impute_knn(all_predictors())

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)


###############
# FIT THE RF MODEL
###############
set.seed(123)
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, kap, f_meas, bal_accuracy)) #select any accuracy measures of your choice


###############
# DISPLAY THE BEST MODELS WITH RESPECITVE PARAMETERS
###############
#Show the best model by kappa statistics
rf_res %>% show_best(metric = "kap", n=15)

#Select the best parameters
rf_best <- 
  rf_res %>% 
  select_best(metric = "kap")


#Plot the ROC_AUC curve
rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(Gender, .pred_female) %>% 
  mutate(model = "Random Forest")

autoplot(rf_auc)


###############
# FIT THE MODEL WITH THE BEST PARAMETERS
###############
# Set up model
last_rf_mod <- 
  rand_forest(mtry = 11, min_n = 13, trees = 1783) %>% 
  set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
  set_mode("classification")

# update workflow
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_mod)

# The final fit
set.seed(123)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(splits)

# View the accuracy of the model
last_rf_fit %>% 
  collect_metrics()

#Visualize the variable importance scores
last_rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 20)

#Plot the ROC_AUC curve
last_rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(Gender, .pred_female) %>% 
  autoplot()

#Generate the confusion matrix
last_rf_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = Gender, estimate = .pred_class)


###############
# FEATURE SELECTION PART 1
###############

#Remove column Smartphone (as it has the lowest importance score)
filtered_phone <- subset(phone, select = -c(Smartphone))

###############
# FEATURE SELECTION PART 2
###############
#Repeat the step to fit and validate the models
set.seed(123)
filtered_splits <- initial_split(filtered_phone, strata = Gender, prop=0.8)
filtered_phone_train <- training(filtered_splits)
filtered_phone_test <- testing(filtered_splits)

#Cross-validation set
set.seed(123)
filtered_val_set <- validation_split(filtered_phone_train, strata = Gender, prop = 0.8)

rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees=tune()) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

rf_recipe <- 
  recipe(Gender ~ ., data = filtered_phone_train) %>%
  step_normalize(all_numeric()) %>%
  step_impute_knn(all_predictors())

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

set.seed(123)
rf_res <- 
  rf_workflow %>% 
  tune_grid(filtered_val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, kap, f_meas, bal_accuracy))

#Show the best parameters
rf_res %>% show_best(metric = "kap", n=15)

#Select the best parameters
rf_best <- 
  rf_res %>% 
  select_best(metric = "kap")

#Plot the ROC_AUC curve
rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(Gender, .pred_female) %>% 
  mutate(model = "Random Forest")
autoplot(rf_auc)

# the final model
last_rf_mod <- 
  rand_forest(mtry = 4, min_n = 9, trees = 497) %>%  #update the best parameters here
  set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
  set_mode("classification")

# Update workflow
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_mod)

# The final fit
set.seed(123)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(filtered_splits)

# View the accuracy of the model
# Unfortunately, the model with subset feature does not perform better
last_rf_fit %>% 
  collect_metrics()

last_rf_fit %>% 
  collect_predictions() %>% 
  roc_curve(Gender, .pred_female) %>% 
  autoplot()

last_rf_fit %>% 
  collect_predictions() %>% 
  conf_mat(truth = Gender, estimate = .pred_class)
