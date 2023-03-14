import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge  import KernelRidge
from sklearn.svm import SVR
import seaborn as sns
import matplotlib.pyplot as plt


#Due to this being a work in progress, there might be a few lines of code with no real purpose as I tend to jump from task to task to focus better, as well as some unfinished comments here and there.
#Sorry for the trouble!

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")

    #demonstrateHelpers(trainDF) #// commented this out for better console readability
    
    #Got the idea from https://www.programiz.com/python-programming/methods/set/intersection
    #print(getAttrsWithMissingValues(trainDF).intersection(getNonNumericAttrs(trainDF)))
    
    #BEGIN: from https://stackoverflow.com/questions/50923707/get-column-name-which-contains-a-specific-value-at-any-rows-in-python-pandas
    #EXPLANATION: goes through the columns of our dataset and retrieves the column names containing the values specified
    #print(trainDF.columns[trainDF.isin(['Ex', 'TA', 'Gd', 'Fa', 'Po']).any()])
    #END: fromhttps://stackoverflow.com/questions/50923707/get-column-name-which-contains-a-specific-value-at-any-rows-in-python-pandas

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    lassoExperiment(trainInput, trainOutput, predictors)
    gbrExperiment(trainInput, trainOutput, predictors)
    kernelExperiment(trainInput, trainOutput, predictors)
    linRidgeExperiment(trainInput, trainOutput, predictors)
    rfExperiment(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)
    
    #numDF = trainDF.select_dtypes(include=['int64', 'float64'])
    
    #correlatedVals = numDF.corr()
    #print(correlatedVals['SalePrice'] >= 0.4)
    
    #plot to show correlation between total property sq.ft and sale price
    #myPlot(trainDF['totalSF'], trainDF['SalePrice'])
 

#got the inspiration from homework 5
def myPlot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x=x, y=y)
    plt.ylabel(y.name, fontsize=10)
    plt.xlabel(x.name, fontsize=10)
    plt.show()

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw06 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)
    
    # Testing lasso regression model, got the idea from: https://vitalflux.com/lasso-ridge-regression-explained-with-python-example/#:~:text=In%20Python%2C%20Lasso%20regression%20can%20be%20performed%20using,therefore%20fewer%20features%20being%20used%20in%20the%20model.
    
def lassoExperiment(trainInput, trainOutput, predictors):
    alg = linear_model.Lasso(alpha = 1.0)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV average score Lasso:", cvMeanScore)
    
def gbrExperiment(trainInput, trainOutput, predictors):
    alg = GradientBoostingRegressor(n_estimators = 100, learning_rate=0.1)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print('CV average score GBR:', cvMeanScore)
    
def kernelExperiment(trainInput,trainOutput, predictors):
    alg = KernelRidge(alpha = 1.0)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print('CV Average Score Kernel Ridge:', cvMeanScore)
    
def linRidgeExperiment(trainInput, trainOutput, predictors):
    alg = Ridge(alpha = 1.0)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print('CV Average Score Linear Ridge:', cvMeanScore)
 
#Got the idea of using random forest from the kaggle main page
def rfExperiment(trainInput, trainOutput, predictors):
    alg = RandomForestRegressor()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print('CV Average Score Random Forest:', cvMeanScore)
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization


'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    
    ordPredictors = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    
    for i in ordPredictors:
        preprocessQual(trainDF, testDF, i)
    
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    '''
    Feature engineering below, created new attributes that would be relevant to our dataset
    '''
    
    #calculating the age of the property at the time of the sale
    trainDF['ageAtSale'] = trainDF['YrSold'] - trainDF['YearBuilt']
    testDF['ageAtSale'] = testDF['YrSold'] - testDF['YearBuilt']
    
    #calculates the total square footage of the property
    trainDF['totalSF'] = trainDF['1stFlrSF'] + trainDF['2ndFlrSF'] + trainDF['TotalBsmtSF'] + trainDF['GarageArea'] + trainDF['GrLivArea'] + trainDF['PoolArea']
    testDF['totalSF'] = testDF['1stFlrSF'] + testDF['2ndFlrSF'] + testDF['TotalBsmtSF'] + testDF['GarageArea'] + testDF['GrLivArea'] + testDF['PoolArea']
    
    #calculates if the property has been remodeled recently
    trainDF['remodelAge'] = trainDF['YrSold'] - trainDF['YearRemodAdd']
    testDF['remodelAge'] = testDF['YrSold'] - testDF['YearRemodAdd']
    
    bathAtts = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
    roomAtts = ['BedroomAbvGr', 'TotRmsAbvGrd', 'KitchenAbvGr', 'Fireplaces']
    locAtts = ['Street', 'Alley']    
    sfAtts = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea','GrLivArea', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', 'ScreenPorch', '3SsnPorch', 'PoolArea', 'MasVnrArea','LotArea', 'totalSF']
    yrAtts = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'ageAtSale', 'remodelAge']
    normalize(trainDF, testDF, sfAtts)
    
    
    fillMissingOrd(trainDF, testDF, 'MSZoning')
    
    preprocessMS(trainDF, testDF, 'MSZoning')
    lotShapeHelper(trainDF, testDF)
    
    numValMissing = getAttrsWithMissingValues(testDF).intersection(getNumericAttrs(testDF))    

    for i in numValMissing:
        fillMissingMean(trainDF, testDF, i)
        
        
    predictors = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GarageArea', 'GarageCars','GrLivArea' , 'OverallQual', 'WoodDeckSF', 'OpenPorchSF','EnclosedPorch', 'ScreenPorch', '3SsnPorch', 'PoolArea', 'MasVnrArea', 'totalSF', 'LotArea', 'MSZoning', 'LotShape'] + ordPredictors + locAtts + bathAtts + roomAtts + yrAtts
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]   
    
    for i in locAtts:
        fillMissingOrd(trainInput, testInput, i)
    nomToNumLocAtts(trainInput, testInput)   

    
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors

    
def preprocessQual(trainDF, testDF, col):

    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v: 0 if v == 'Po' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v: 1 if v == 'Fa' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v: 2 if v == 'TA' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v: 3 if v == 'Gd' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v: 4 if v == 'Ex' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode()[0])
    
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v: 0 if v == 'Po' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v: 1 if v == 'Fa' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v: 2 if v == 'TA' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v: 3 if v == 'Gd' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v: 4 if v == 'Ex' else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mode()[0])
    
def preprocessMS(trainDF, testDF, col):
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  7 if v=='FV' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  6 if v=='RM' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  5 if v=='RP' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  4 if v=='RL' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  3 if v=='I' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  2 if v=='RH' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  1 if v=='C (all)' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  0 if v=='A' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode()[0])

    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  7 if v=='FV' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  6 if v=='RM' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  5 if v=='RP' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  4 if v=='RL' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  3 if v=='I' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  2 if v=='RH' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  1 if v=='C (all)' else v)
    testDF.loc[:, col] = testDF.loc[:, col].map(lambda v:  0 if v=='A' else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mode()[0])
    
    
def lotShapeHelper(trainDF, testDF):
    trainDF['LotShape'].replace({'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3}, inplace=True)
    testDF['LotShape'].replace({'IR3':0, 'IR2': 1, 'IR1': 2, 'Reg': 3}, inplace=True)
    
    
def nomToNumLocAtts(trainDF, testDF):
    trainDF['Street'].replace({'Grvl': 1, 'Pave': 0}, inplace = True)
    testDF['Street'].replace({'Grvl': 1, 'Pave': 0}, inplace = True)
    trainDF['Alley'].replace({'Grvl':2, 'Pave': 1, 'NA':0}, inplace = True)
    testDF['Alley'].replace({'Grvl':2, 'Pave': 1, 'NA':0}, inplace = True)
    
def fillMissingOrd(trainDF, testDF, col):
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v: trainDF.loc[:, col].mode().loc[0] if v == 'Na' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].map(lambda v:  trainDF.loc[:, col].mode().loc[0] if v=='None' else v)
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mode().loc[0])
    
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=='NA' else v)
    testDF.loc[:,col] = testDF.loc[:,col].map(lambda v: testDF.loc[:, col].mode().loc[0] if v=='None' else v)
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mode().loc[0])
        
    
    # Helper function to fill missing numerical values with the mean
def fillMissingMean(trainDF, testDF, col):
    trainDF.loc[:, col] = trainDF.loc[:, col].fillna(trainDF.loc[:, col].mean())
    testDF.loc[:, col] = testDF.loc[:, col].fillna(testDF.loc[:, col].mean())

    
def standardize(trainDF, testDF, col):
    trainDF.loc[:,col] = trainDF.loc[:,col].apply(lambda row: (row-trainDF.loc[:,col].mean())/trainDF.loc[:,col].std(), axis=1)
    testDF.loc[:,col] = testDF.loc[:,col].apply(lambda row: (row-testDF.loc[:,col].mean())/testDF.loc[:,col].std(), axis=1)
    
def normalize(trainDF, testDF, col):
    trainDF.loc[:,col] = trainDF.loc[:,col].apply(lambda row:(row- trainDF.loc[:,col].min())/((trainDF.loc[:, col].max())- (trainDF.loc[:,col].min())), axis =1)
    testDF.loc[:,col] = testDF.loc[:,col].apply(lambda row:(row- testDF.loc[:,col].min())/((testDF.loc[:, col].max())- (testDF.loc[:,col].min())), axis =1)
    


# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

