# Short description of df column:
  df['column_name].describe()
# Of the whole dataframe:
  df.describe()
   
# One-hot encode columns
  df = pd.get_dummies(df, columns= ['col1', 'col2', '...'])
 
# Correlation Heatmap
  import matplotlib.pyplot as plt
  import seaborn as sns
  corrmat = df.corr()
  f, ax = plt.subplots(figsize = (7,7))
  sns.set(font_scale=0.9)
  sns.heatmap(corrmat, vmax=0.8, square=True, annot=True, fmt='.2f', cmap= "winter")
  plt.show()

## Copyable Code for Linear Regression:
  # Simple Linear Regression
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LinearRegression
  X = df.drop('column1', 'column2', '...', axis = 1) #usually just drop the target column, but can drop other unwanted columns as well
  Y = df['target_column']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  linear_model = LinearRegression()
  linear_model.fit(X_train, Y_train)
  
    # Lasso Linear Regression (Prevents Overfitting and improves model)
    from sklearn.linear_model() import Lasso
    lasso_model = Lasso(alpha=0.5, normalize = True) #increasing alpha may improve model by forcing important predictors to have more influence
    lasso_model.fit(X_train, y_train)
     
    # Ridge Linear Regression (Used to tackle multicollinearity and if a dataset has a large amount of predictors)
    from sklearn.linear_model() import Ridge   
    ridge_model = Ridge(alpha=0.5, normalize = True) #increasing alpha may improve model by forcing important predictors to have more influence
    ridge_model.fit(X_train, y_train)
     
    # SVM Linear Regression (Used for a small dataset, heavily penalizes outliers)
    from sklearn.svm import SVR
    regression_model = SVR(kernel = 'linear', C = 1.0) #C is the penalty parameter, can change this parameter as needed.
    regression_model.fit(X_train, y_train)
    
  # R2 score of the training data
  linear_model.score(X-train,Y-train)
  # R2 score of the test data
  linear_model.score(y_predict,y_test)
  
  # see all the indiviual weights of predictors
  predictors = X_train.columns
  coef = pd.Series(linear_model.coef_,predictors).sort_values()
  print(coef)
     
  # visually looking at the predicted value by the model compared to the actual value
  y_predict = linear_model.predict(X_test)
  %pylab inline
  pylab.rcParams['figure.figsize'] = (15,6)
  plt.plot(y_predict, label = 'Predicted')
  plt.plot(y_test.values, label = 'Actual')
  plt.ylabel('target_columnName')
  plt.legend()
  plt.show()

# Gradient Boosting Regression Example:
  from sklearn.ensemble import GradientBoostingRegressor
  params = {'n_estimators': x, 'max_depth': x, 'min_samples_split': x, 'learning_rate': x, 'loss': 'ls'} #enter in values for x
  gbr_model = GradientBoostingRegressor(**params)
  gbr_model.fit(X_train, Y_train)
     
  # get r2 score with previous code
     
    # Hyperparamter tuning with commonly used values
      from sklearn.model_selection import GridSearchCV
      num_estimators = [100, 200, 500]
      learn_rates = [0.01, 0.02, 0.05, 0.1]
      max_depths = [4, 6, 8]
      param_grid = {'num_estimators': num_estimators, 'learn_rates': learn_rates, 'max_depths': max_depths}
      grid_search = GridSearchCV(GradientBoostingRegressor(min_samples_split = 2, loss = 'ls'), param_grid, cv = 3, return_train_score = True)
      grid_search.fit(X_train, Y_train)
      grid_search.best_params_
      # go back to the GBR code and use the best params given
      
  
