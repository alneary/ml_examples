# Databricks notebook source
# MAGIC %md
# MAGIC # Supervised Models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Trees

# COMMAND ----------

#3 Fitting the Decision Tree Regression Model to the dataset
# Create the Decision Tree regressor object here
from sklearn.tree import DecisionTreeRegressor
#DecisionTreeRegressor class has many parameters. Input only #random_state=0 or 41.
model = DecisionTreeRegressor(random_state = 0)
# Fitting the Decision Tree Regression model to the data
model.fit(x_train, y_train) 

# Predicting the target values of the test set
y_pred = model.predict(x_test)
# RMSE (Root Mean Square Error)
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)), '.3f'))
print("\nRMSE: ", rmse)

from sklearn.tree import export_graphviz  
# export the decision tree model to a tree_structure.dot file 
# paste the contents of the file to webgraphviz.com
export_graphviz(model, out_file ='tree_structure.dot', 
               feature_names =['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)'])  

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor 
  
# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0) 
  
# fit the regressor with X and Y data
regressor.fit(X, y)

# predicting a new value
  
# test the output by changing values, like 3750
y_pred = regressor.predict([[3750]])
  
# print the predicted price
print("Predicted price: % d\n"% y_pred) 

# import export_graphviz
from sklearn.tree import export_graphviz 
  
# export the decision tree to a tree.dot file
# for visualizing the plot easily anywhere
export_graphviz(regressor, out_file ='tree.dot',
               feature_names =['Production Cost']) 

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### iModels

# COMMAND ----------

from sklearn.model_selection import train_test_split
from imodels import get_clean_dataset, HSTreeClassifierCV # import any imodels model here

# prepare data (a sample clinical dataset)
X, y, feature_names = get_clean_dataset('csi_pecarn_pred')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

# fit the model
model = HSTreeClassifierCV(max_leaf_nodes=4)  # initialize a tree model and specify only 4 leaf nodes
model.fit(X_train, y_train, feature_names=feature_names)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)
print(model) # print the model

# COMMAND ----------

# MAGIC %md
# MAGIC #### Lasso

# COMMAND ----------

# define model
lasso_model = Lasso().fit(X_train, y_train)
#get intercept
print(lasso_model.intercept_)
#get coefficients
print(lasso_model.coef_)
#predict on test and get results
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#get r2 score
r2_score(y_test, y_pred)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Tuning a Lasso model for optimal selection of lambda/c

# COMMAND ----------

#version a
# define model
model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X, y)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# COMMAND ----------

#version b
lasso_cv_model = LassoCV(alphas = np.random.randint(0,1000,100), cv = 10, max_iter = 100000).fit(X_train,y_train)
#get alpha
lasso_cv_model.alpha_
#now create the optimized model
lasso_tuned = Lasso().set_params(alpha = lasso_cv_model.alpha_).fit(X_train,y_train)

y_pred_tuned = lasso_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred_tuned))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression using scikit learn

# COMMAND ----------

#Linear Regression
from sklearn.linear_model import LinearRegression
lModel = LinearRegression()
lModel.fit(X_train,Y_train)
print_score(lModel)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression using statsmodels

# COMMAND ----------

import scipy.stats as sps
import statsmodels.api as sm
import statsmodels.formula.api as smf
#single variable regression
bm_mod = smf.ols('body_mass_g ~ flipper_length_mm', data=penguins)
bmf = bm_mod.fit()
bmf.summary()

#multi variate regression 
bm_mod = smf.ols('mass ~ flipper + species', data=penguin_std)
bmf = bm_mod.fit()
bmf.summary()

#multi with interaction term (flipper * species expands to flipper + species + flipper:species)
bm_mod = smf.ols('mass ~ flipper * species', data=penguin_std)
bmf = bm_mod.fit()
bmf.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Diagnostic plots for Linear Regression output of statsmodels
# MAGIC Assumptions need to be checked, specifically for the residuals. 
# MAGIC * Residuals vs fitted values
# MAGIC * Residual Q-Q plots

# COMMAND ----------

#this function plots both from above
def plot_lm_diag(fit):
    "Plot linear fit diagnostics"
    sns.regplot(x=fit.fittedvalues, y=fit.resid)
    plt.xlabel('Fitted')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')
    plt.show()

    sm.qqplot(fit.resid, fit=True, line='45')
    plt.title('Residuals')
    plt.show()
    
plot_lm_diag(bmf)
    


# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest Regressor

# COMMAND ----------

#Random Forest
from sklearn.ensemble import RandomForestRegressor
m = RandomForestRegressor()
m.fit(X_train, Y_train)
print_score(m)

# COMMAND ----------

# MAGIC %md
# MAGIC #### KNN Regressor

# COMMAND ----------

#KNN
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor()
knnr.fit(X_train,Y_train)
print_score(knnr)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gradient Boost

# COMMAND ----------

#Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor()
GBR.fit(X_train,Y_train)
print_score(GBR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification
# MAGIC 
# MAGIC All of these include pipelines that perform some sort of dimension reduction

# COMMAND ----------

#packages used
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, make_scorer
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Trees

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier, plot_tree
#note the regularization of the max depth of 2 below
dtree = DecisionTreeClassifier(max_depth=2)
dtree.fit(train_x, train_y)
print(accuracy_score(train_y, dtree.predict(train_x)))
plot_tree(dtree, feature_names=feat_cols, impurity=False, class_names=['Deny', 'Admit'], label='root')


# COMMAND ----------

#unlimited, unregularized decision tree -> prone to overfitting
uldtree = DecisionTreeClassifier()
uldtree.fit(train_x, train_y)

uldtree = DecisionTreeClassifier()
uldtree.fit(train_x, train_y)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Lasso

# COMMAND ----------

from sklearn.linear_model import LogisticRegressionCV

lasso_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('scaler', StandardScaler()),
    ('classify', LogisticRegressionCV(penalty='l1', cv=4, solver='saga', max_iter=500, n_jobs=4))
])

lasso_pipe.fit(train_x, train_y)
lasso_pipe.named_steps['classify'].coef_
print("Training data accuracy", accuracy_score(train_y, lasso_pipe.predict(train_x)))
print("Test data accuracy", accuracy_score(test_y, lasso_pipe.predict(test_x)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

#Fit pipeline
log_pipe = Pipeline([
    ('standardize', StandardScaler()),
    ('classify', LogisticRegression(penalty='none'))
])

#train pipeline
log_pipe.fit(train_x, train_y)

#get coeffs
log_pipe.named_steps['classify'].coef_

#print metrics
print("accuracy score:", accuracy_score(test_y, log_pipe.predict(test_x)))
print("precision score:", precision_score(test_y, log_pipe.predict(test_x)))
print("recall score:", recall_score(test_y, log_pipe.predict(test_x)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Multinomial Naive Bayes

# COMMAND ----------

from sklearn.naive_bayes import MultinomialNB
bayes_pipe = Pipeline([
    ('word_count', CountVectorizer()),
    ('classify', MultinomialNB())
])
bayes_pipe.fit(train['text'], train['category'])
#Report its training and test accuracy and add to dataframe for later use
print('Training accuracy: ', accuracy_score(train['category'], bayes_pipe.predict(train['text'])))
print('Test accuracy: ', accuracy_score(test['category'], bayes_pipe.predict(test['text'])))
clsresults = pd.DataFrame([['bayes', accuracy_score(train['category'], bayes_pipe.predict(train['text'])),
    accuracy_score(test['category'], bayes_pipe.predict(test['text']))]], 
    columns=['type','train_result','test_result'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest Classifier

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('classify', RandomForestClassifier(max_depth = 10))    
])

rf_pipe.fit(train_x, train_y)


print("Training data accuracy", accuracy_score(train_y, rf_pipe.predict(train_x)))
print("Test data accuracy", accuracy_score(test_y, rf_pipe.predict(test_x)))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Ridge Regression Classifier

# COMMAND ----------

from sklearn.linear_model import LogisticRegressionCV

reg_pipe = Pipeline([
    ('standardize', StandardScaler()),
    ('classify', LogisticRegressionCV())
])
#train it
reg_pipe.fit(train_x, train_y)
#extract coefficients
reg_model = reg_pipe.named_steps['classify']
reg_model.coef_

#determine what regularization strength it learned
reg_model.C_

#training accuracy
accuracy_score(train_y, reg_pipe.predict(train_x))

#apply the model to untransformed variables -> it will have no idea what to do and accuracy will bomb
accuracy_score(train_y, reg_model.predict(train_x))



# COMMAND ----------

# MAGIC %md
# MAGIC #### KNN Classifier

# COMMAND ----------

from sklearn.naive_bayes import MultinomialNB
#k=5
knn5_pipe = Pipeline([
    ('word_vec', TfidfVectorizer()),
    ('knn', KNeighborsClassifier(n_neighbors = 5))
])
knn5_pipe.fit(train['text'], train['category'])
print("Training set accuracy of classifier: ", knn5_pipe.score(train['text'], train['category']))
print("Test set accuracy of classifier: ", knn5_pipe.score(test['text'], test['category']))

to_append = ['knn w/ k=5', knn5_pipe.score(train['text'], train['category']), knn5_pipe.score(test['text'], test['category'])]
df_len=len(clsresults)
clsresults.loc[df_len] = to_append

# COMMAND ----------

# MAGIC %md
# MAGIC #### KNN with grid search
# MAGIC transforming text data into a dfidf matrix (because the example was a set of words)

# COMMAND ----------

from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn_search = GridSearchCV(KNeighborsClassifier(), {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13]    
}, scoring=make_scorer(accuracy_score))
knn_results = knn_search.fit(X_train, Y_train)
print("The best resulting score is", knn_results.best_score_)
print("The best param is ", knn_results.best_params_)
#test dataset...
knn_results.score(X_test, Y_test)
#append to results dataset
to_append = ['knn w/ grid', knn_results.score(X_train, Y_train),knn_results.score(X_test, Y_test)]
df_len=len(clsresults)
clsresults.loc[df_len] = to_append

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Unsupervised

# COMMAND ----------

# MAGIC %md
# MAGIC #### Kmeans Clustering

# COMMAND ----------

from sklearn.cluster import KMeans
#see below for the tfid vectorizer in other section
cluster_pipe = Pipeline([
    ('vectorize', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('cluster', KMeans(5))
])
fit_clusters = cluster_pipe.fit(articles['text'])

article_clusters = pd.DataFrame(cluster_pipe.predict(articles['text']), columns=['cluster'])
# The output of predict is cluster numbers, we will use these as data points.

#check the values
article_clusters['cluster'].value_counts()

# COMMAND ----------

#Plot the cluster/category in a facet grid bar plot. Show the distribution of the category by each of the clusters. First we have to join the data together.
clustered_categories = pd.concat([article_clusters, articles], axis=1)

g = sns.FacetGrid(clustered_categories, col='cluster')
g.map(sns.countplot, 'category', order =['business','entertainment','politics','sport','tech'])
g.set_xticklabels(rotation=30)
plt.show

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Others

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimension Reduction

# COMMAND ----------

# MAGIC %md
# MAGIC ### Truncated SVD

# COMMAND ----------

svd_pipe = Pipeline([
    ('word_vec', TfidfVectorizer()),
    ('svd', TruncatedSVD(8))
])
svd_pipe.fit(train['text'])
text_vectors = svd_pipe.transform(train['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Matrix Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ##### TFIDF matrix transformation

# COMMAND ----------

tfidf = TfidfVectorizer(stop_words='english')
X_train = tfidf.fit_transform(train['text'])
Y_train = train['category']
X_test = tfidf.transform(test['text'])
Y_test = test['category']
print(X_train.shape)

# COMMAND ----------


