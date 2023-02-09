# Databricks notebook source
# MAGIC %sh apt-get install -y graphviz

# COMMAND ----------

# MAGIC %md
# MAGIC #### Purpose/Context
# MAGIC This workbook focuses on building a predicitive model using the Seoul Bike Dataset. A brief EDA will take place before model development.
# MAGIC 
# MAGIC ##### Problem Statement
# MAGIC To build a model from the existing data to predict bike rentals. In other words, if we had future weather information, how could we predict bike rental counts.

# COMMAND ----------

# MAGIC %pip install --upgrade imodels
# MAGIC %pip install --upgrade dtreeviz
# MAGIC %pip install --upgrade skompiler

# COMMAND ----------

# MAGIC %run ./0_data_preprocessing_seoul_bike_project

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

import plotly.express as px
import plotly.figure_factory as ff

import warnings
warnings.simplefilter("once")

%matplotlib inline
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Ingestion
# MAGIC Read in the cleaned dataset and filter it to only those records that have a valid bike rental count (to reduce noise).

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Read in file

# COMMAND ----------

#read in the cleaned parquet file, which was created in a previous notebook
df_raw = pd.read_parquet('seoul_bike_dataset_final')

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA

# COMMAND ----------

# MAGIC %md
# MAGIC #### Outcome variable - rented bike count

# COMMAND ----------

#first look at the distribution of our outcome variable
sns.displot(x = 'rented_bike_count', data = df_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC Remarks: 
# MAGIC - 0 is the mode
# MAGIC - not normally distributed, 
# MAGIC - positively skewed/long right tail and 
# MAGIC - bound by zero, with a high count of 0 rentals. 
# MAGIC 
# MAGIC This directly affects the types of models we should use, as a negative prediction makes no sense and the errors are (most likely) not normally distributed.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Functioning Day
# MAGIC The definition of this variable leads to some concern about whether these observations should be included in the analysis.

# COMMAND ----------

df_raw.groupby('functioning_day')['rented_bike_count'].agg(['count','mean', 'min', 'max'])

# COMMAND ----------

sns.displot(x = 'rented_bike_count', data = df_raw, hue='functioning_day')

# COMMAND ----------

#filter to only those records representative of a functioning day
df = df_raw.loc[df_raw['functioning_day'] == "Yes"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Other EDA

# COMMAND ----------

# MAGIC %md
# MAGIC #### Examine the correlation (linear relationship) between the continous vars

# COMMAND ----------

sns.pairplot(df[['rented_bike_count', 'temperature','humidity','wind_speed','visability','dew_point','solar_radiation','rainfall']])

# COMMAND ----------

#heatmap
sns.heatmap(df.corr())

# COMMAND ----------

# MAGIC %md
# MAGIC Finding: There is a great degree of correlation between dew_point and (temperature and humidity). We should consider removing this variable

# COMMAND ----------

# MAGIC %md
# MAGIC #### Categorical Vars

# COMMAND ----------

fig, axs= plt.subplots(figsize=(20, 8), ncols=3)
sns.boxplot(x='rented_bike_count', y= 'hour',  data = df, ax=axs[0])
sns.boxplot(x='rented_bike_count', y='season', data = df, ax=axs[1])
sns.boxplot(x='rented_bike_count', y='holiday', data = df, ax=axs[2])


# COMMAND ----------

# MAGIC %md
# MAGIC Findings:
# MAGIC 
# MAGIC - There appear to be spikes throughout the day, occuring around the morning commute and then again in the evening. 
# MAGIC - Winter has the lowest amount of rentals with summer being generally greater. Need to take winter olympics into consideration here though.
# MAGIC - Rentals are typically lower on holidays indicating that perhaps folks are using these bikes for business purposes.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Variable creation and transformation

# COMMAND ----------

#Transformation
###clean up cat var for holiday so that we can create dummies later
df['holiday'] =df['holiday'].replace(' ', '_', regex = True)

#Creation
###next create a variable to flag if the date is a weekday
dates = pd.to_datetime(df['date'])
df.loc[dates.dt.dayofweek > 4, 'weekend']='Yes'
df.loc[dates.dt.dayofweek <= 4, 'weekend']='No'

###next create a hour category to assign an hour to either peak, off-peak or regular hours
df.loc[(df['hour'].isin(['17','18','19'])), 'hour_cat']='Evening_Peak'
df.loc[(df['hour'].isin(['8'])), 'hour_cat']='Morning_Peak'
df.loc[(df['hour'].isin(['2','3','4','5','6'])), 'hour_cat']='Off_Peak'
df.loc[(df['hour'].isin(['7','9','10','11','12','13','14','15','20','21','22','23','0','1'])), 'hour_cat']='Regular'
df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Modeling

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dummy variables

# COMMAND ----------

#add dummies for categorical variables
df = df.join(pd.get_dummies(df['season'], drop_first = True))
df = df.join(pd.get_dummies(df['holiday'], drop_first = True))
df = df.join(pd.get_dummies(df['weekend'], drop_first = True, prefix = 'weekend'))
df = df.join(pd.get_dummies(df['hour'], drop_first = True, prefix = 'hour'))
df = df.join(pd.get_dummies(df['hour_cat'], drop_first = True, prefix = 'hour_cat'))

# COMMAND ----------

df.columns

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare data for modeling and evaluation
# MAGIC We will use a 80/20 train/test split of the data. Since we are performing model selection we will create a validation set so that we can preserve the test dataset til we select our final model. For simplicity we will just create one split of the training set at 20%. <br>
# MAGIC Full dataset (n=8465) <br>
# MAGIC <u> For model selection 
# MAGIC * Training dataset (n=5417) ~ 80% of full training set or ~64% of full dataset
# MAGIC * Validation set (n=1355) ~ 20% of full training set or ~16% of full dataset
# MAGIC --------------------------------------------------------------------------------
# MAGIC For model assessment
# MAGIC   * (Full) training dataset (n=6772) -> 80%
# MAGIC   * Test dataset (n=1693) -> 20%
# MAGIC   
# MAGIC   
# MAGIC   
# MAGIC To Do -> 
# MAGIC 1. only include necessary vars
# MAGIC 2. Stratify on important vars
# MAGIC 3. Document why you chose the predictors you did

# COMMAND ----------

# MAGIC %md
# MAGIC #### Specify predictor variables and outcome

# COMMAND ----------

#specify the various features to use in the modeling
weather_vars = ["temperature",  "wind_speed", "visability", "solar_radiation", "rainfall", "snowfall"]
hour_vars = ["hour_1", "hour_2", "hour_3","hour_4", "hour_5", "hour_6", "hour_7", "hour_8", "hour_9", "hour_10", "hour_11", "hour_12", "hour_13", "hour_14", "hour_15", "hour_16", "hour_17", "hour_18", "hour_19", "hour_20", "hour_21", "hour_22", "hour_23"]
#hour_vars = ['hour_cat_Morning_Peak', 'hour_cat_Regular', 'hour_cat_Off_Peak']
other_x_vars = ["weekend_Yes", "Spring", "Summer", "Winter", "No_Holiday"]
#select features for model
x_vars = weather_vars + other_x_vars + hour_vars
y_var = ['rented_bike_count']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create actual splits

# COMMAND ----------

#first create the full training and test split
X_full_train, X_test, y_full_train, y_test = train_test_split(df[x_vars], df[y_var], test_size=0.2, random_state=41)
#next, using the training set, create a validation split off
X_train, X_valid, y_train, y_valid = train_test_split(X_full_train[x_vars], y_full_train[y_var], test_size=0.2, random_state=41)
X_train.head()

# COMMAND ----------

print("X Vars:", X_full_train.shape, X_train.shape, X_valid.shape,  X_test.shape)
print("Y Vars:",  y_full_train.shape, y_train.shape, y_valid.shape, y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC #### General functions used

# COMMAND ----------

def score_model(actual, predicted, model_name, data_type='general'):
    mae = mean_absolute_error(actual,predicted)
    mse = mean_squared_error(actual,predicted) 
    rmse = np.sqrt(mse)
    r2_score(actual,predicted)
    print (data_type ,' data')
    print('mean absolute error is  :',mae)
    print('mean squared error is  :',mse)
    print('Root mean squared error is  :', rmse)
    print("R2 score is  :",r2_score(actual,predicted))
    return pd.DataFrame([[model_name, data_type, mae, mse, rmse, r2_score]], 
                            columns = ['model', 'data_type', 'mae','mse','rmse','r2_score'])
    
def plot_lm_diag(fit):
    "Plot fit diagnostics"
    sns.regplot(x=fit.fittedvalues, y=fit.resid)
    plt.xlabel('Fitted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')
    plt.show()
    
    sm.qqplot(fit.resid, fit = True, line='45')
    plt.title('Residuals')
    plt.show()    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear models

# COMMAND ----------

# MAGIC %md
# MAGIC ### OLS
# MAGIC Refresher on model assumptions:
# MAGIC 1. Linearity -> Relationship between X and mean of Y is linear (scatterplots)
# MAGIC 2. Independence of Errors -> No relationship between residuals and Y variable (plot residuals versus fit values)
# MAGIC 3. Normalit of Errors -> Residuals must be normally distributed (Q-Q plot)
# MAGIC 4. Homoschedasticity (equal variances of the residuals) -> variance of residuals should be the same for all values of X (this is where we run into issues with the count outcome variable) (plot resids versus fits to ensure no pattern)
# MAGIC 
# MAGIC We already know MLR isn't the best option because of the skewedness and long tail of the outcome variable's values. Let's look at what this looks like

# COMMAND ----------

ols_model = sm.OLS(y_train[y_var],X_train[x_vars])
ols_fit = ols_model.fit()
#print("Parameter coefficients", ols_fit.params)
ols_valid_results = score_model(y_valid, ols_fit.predict(X_valid), 'ols', 'valid')
ols_train_results = score_model(y_train, ols_fit.predict(X_train), 'ols', 'train')
ols_fit.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Now diagnose the model and see what the fit looks like

# COMMAND ----------

plot_lm_diag(ols_fit)

# COMMAND ----------

# MAGIC %md
# MAGIC Diagnosing this model: <br>
# MAGIC First, a major issue here is that we have negative values predicted for the bike rentals, something that isn't feasible. Since the bike rental count is actually a count variable, we need to examine models that assume a poisson distribution. <br>
# MAGIC Next, this model is showing signs that the assumptions of linear regression are violated as evidenced by the fanning out in the fit/resid plot and the curling out towards the extreme quartiles in the QQ plots. This indicates that the errors aren't independent and there could be additional correlations not identified. A strong pattern in the residuals indicates non-linearity in the data. A linear model probably isn't ideal in this case.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Poisson Regression
# MAGIC Poisson regression has a skew, discrete distribution and restricts predicted values to positive only.
# MAGIC 
# MAGIC Differences in poisson:
# MAGIC - errors follow a poisson distribution, not a normal one like in OLS
# MAGIC - models ln(Y) as a linear function of the coefficients.
# MAGIC - assumes that the mean and variance of the errors are equal, if the variance is much larger than the mean we can look into the over-dispersed Poisson model (estimating how much larger the variance is than the mean) or we could consider the negative binomial (which is a form of the poisson where the distributions parameter is a random variable).

# COMMAND ----------

poisson_model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.Poisson()).fit()
poisson_valid_results = score_model(y_valid, poisson_model.predict(sm.add_constant(X_valid)), 'poisson', 'valid')
poisson_train_results = score_model(y_train, poisson_model.predict(sm.add_constant(X_train)), 'poisson', 'train')
poisson_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Examine how the predicted values are measuring up compared to actual values.

# COMMAND ----------

#Show scatter plot of Actual versus Predicted counts
plt.clf()
fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for Poisson Model')
plt.scatter(x=poisson_model.predict(sm.add_constant(X_valid)), y=y_valid, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Results:
# MAGIC 
# MAGIC These results don't look terrible, but they leave room for improvement. We expect the results to fan out because of the way the data are distributed.
# MAGIC 
# MAGIC Recall that the assumption of this model is that the mean = variance. The pearson chi square of this model is quite large, much larger than the chi square statistic would be for a dataset this size. We can look into the negative binomial model in future steps.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Negative Binomial Regression

# COMMAND ----------

negbin_model = sm.GLM(y_train, sm.add_constant(X_train), family=sm.families.NegativeBinomial()).fit()

negbin_valid_results = score_model(y_valid, negbin_model.predict(sm.add_constant(X_valid)), 'neg_bin', 'valid')
negbin_train_results = score_model(y_train, negbin_model.predict(sm.add_constant(X_train)), 'neg_bin', 'train')
negbin_model.summary()

# COMMAND ----------

plt.clf()
fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts for Neg Binomial Model')
plt.scatter(x=negbin_model.predict(sm.add_constant(X_valid)), y=y_valid, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Remarks:
# MAGIC Negative Binomial model fits the problem much better (avoiding the mean = variance assumption).
# MAGIC 
# MAGIC Next steps should involve working on improving this model:
# MAGIC - is parsimoneous model best here?
# MAGIC - Creating better features, reducing the number of features in the model.  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tree Based Algorithms

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree (Single)

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor, export_text
#DecisionTreeRegressor class has many parameters. Input only #random_state=0 or 41.
dt_model = DecisionTreeRegressor(random_state = 0, max_depth=5)
dt_fit = dt_model.fit(X_train, y_train) 

dt_valid_results = score_model(y_valid, dt_model.predict(X_valid), 'dt', 'valid')
dt_train_results = score_model(y_train, dt_model.predict(X_train), 'dt', 'train')

# COMMAND ----------

r = export_text(dt_model, feature_names=x_vars)
print(r)


# COMMAND ----------

import dtreeviz
from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
from skompiler import skompile

# COMMAND ----------

def view_tree(model, tree_num, x, y):
    #specify nclasses = 1 when continuous variable
    dt = extract_sklearn_tree_from_figs(model, tree_num = tree_num, n_classes = 1)
    print(dir(dt))
    shadow_dtree = ShadowSKDTree(dt, x, y, x.columns, 'title')
    
    viz_model = dtreeviz.model(shadow_dtree, x, y, x.columns, "title")
    displayHTML(viz_model.view(scale = 1.5).svg())
        

# COMMAND ----------

# MAGIC %md
# MAGIC ### Greedy Tree 

# COMMAND ----------

#iModels
from imodels import GreedyTreeRegressor, FIGSRegressorCV

# fit the model
gt_model = GreedyTreeRegressor()  
gt_model.fit(X_train, y_train, feature_names=x_vars)   # fit model

#gt_model.fit(X_train, y_train, feature_names=x_vars)   # fit model

gt_valid_results = score_model(y_valid, gt_model.predict(X_valid), 'gt', 'valid')
gt_train_results = score_model(y_train, gt_model.predict(X_train), 'gt', 'train')

# COMMAND ----------

gt_model

# COMMAND ----------

# MAGIC %md
# MAGIC ### FIGS

# COMMAND ----------

warnings.filterwarnings("ignore")
figs_model = FIGSRegressorCV(scoring = 'neg_mean_squared_error')
figs_fit = figs_model.fit(X_train, y_train['rented_bike_count']) 
figs_valid_results = score_model(y_valid, figs_model.predict(X_valid), 'figs', 'valid')
figs_train_results = score_model(y_train, figs_model.predict(X_train), 'figs', 'train')

# COMMAND ----------

model_figs = figs_model.figs
model_figs

# COMMAND ----------

view_tree(model_figs, 0, X_valid, y_valid.to_numpy())

# COMMAND ----------

#this is the code to extract the rules/logic, I couldn't get it to work, but perhaps there will be better luck in the future, or at least better doc
#https://github.com/csinva/imodels/blob/master/notebooks/FIGS_viz_demo.ipynb
#dt = extract_sklearn_tree_from_figs(model_figs, tree_num = 0, n_classes = 1)
#expr = skompile(dt.score, X_valid.columns)
#print(expr.to('sqlalchemy/sqlite', component = 1, assign_to='tree_0'))
#print(expr.to('python/code')

# COMMAND ----------

ols_train_results

# COMMAND ----------

model_results = pd.concat([ols_train_results, ols_valid_results, negbin_train_results, negbin_valid_results, poisson_train_results, poisson_valid_results, dt_train_results, dt_valid_results, gt_train_results, gt_valid_results, figs_train_results, figs_valid_results], ignore_index=True)

train_model_results = model_results.loc[model_results['data_type']=='train']
valid_model_results = model_results.loc[model_results['data_type']=='valid']


# COMMAND ----------

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot('model','rmse', data = model_results, hue='data_type')

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion/ Next Steps:
# MAGIC - Generally, our models are at best roughly 250 bikes off, on average, of predicting the number of bikes rented. At worst, roughly 400 when looking at a hold-out validation set. This problem shows the potential for solving the problem with additional investment of a small amount of tuning of these models.
# MAGIC - In comparison to general linear models, the more advanced tree-based models show greater performance. Caution should be taken when interpreting these results as these trees are un-pruned. Pruning should be considered as a next step.
# MAGIC - Across all sets of models (excluding OLS due to its validity), the most important features were temp, solar radiation, and various hours of the day. GLM showed that holiday/no holiday was a major determination as well. We could use this information to engineer better features for the time spikes and also create a more parsimoneous model using fewer variables.
# MAGIC - We've kept a hold out test set of the data that we can use for evaluating our final model, once we've performed additional tuning.
# MAGIC - With respect to GLM consider adding a penalty to the model for regularization.

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC import dtreeviz
# MAGIC from dtreeviz.models.sklearn_decision_trees import ShadowSKDTree
# MAGIC from imodels.tree.viz_utils import extract_sklearn_tree_from_figs
# MAGIC 
# MAGIC def view_tree(model, tree_num, x, y):
# MAGIC     dt = extract_sklearn_tree_from_figs(model, tree_num = tree_num, n_classes = 0)
# MAGIC     shadow_dtree = ShadowSKDTree(dt, x, y, x.columns, 'title')
# MAGIC     
# MAGIC     viz_model = dtreeviz.model(shadow_dtree, x, y, x.columns, "title")
# MAGIC     displayHTML(viz_model.view(scale = 1.5).svg())
# MAGIC     
# MAGIC     view_tree(model_figs, 1, X_valid, y_valid)
