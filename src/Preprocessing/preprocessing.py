# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')

import utils as u
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# Description of this script:
# main function calling the sub preprocessing function for the dataset selected
# subfunctions applying preprocessing (ex: one hot encoding, dropping etc..)


def main_preprocessing_from_name (df, conf):
    """
    Main Preprocessing function: it launches the correct function in order to preprocess the selected dataset
    Args:
        df: Dataframe
        conf: Conf file

    Returns: Preprocessed Dataframe

    """

    dict_function_preprocess = {'bank': 'preprocessing_for_bank',
                                'orfee': 'preprocessing_for_orfee',
                                'diabetic': 'preprocessing_for_diabetic'}

    selected_dataset = conf['selected_dataset']
    function_preprocess = globals()[dict_function_preprocess[selected_dataset]]
    logger.info('Beginning of preprocessing function: ' + dict_function_preprocess[selected_dataset] )
    df_preprocessed = function_preprocess(df, conf)
    logger.info('End of preprocessing function: ' + dict_function_preprocess[selected_dataset] )

    return df_preprocessed


def preprocessing_for_bank(df, conf):
    """
    Preprocessing for the BANKING dataset
    Args:
        df: Banking dataset
        conf:  conf file

    Returns: Preprocessed Banking Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # ordinal encoding elevel, car, zipcode
    # drop id

    logger.debug('Cleaning Output')

    # Cleaning Output:
    df['deposit_subscription'] = df['deposit_subscription'].map({'yes': 1, 'no': 0})

    logger.debug('Mapping Values')
    #Cleaning Other fields:
    for col in  ['loan','housing','default']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    df["month"] = df["month"].map({'jan': 1, 'feb': 2,'mar': 3, 'apr': 4,'may': 5, 'jun': 6,'jul': 7, 'aug': 8,'sep': 9, 'oct': 10,'nov': 11, 'dec': 12})

    logger.debug('Ordinal Encoding')
    # Ordinal encoding
    cols = ['poutcome', 'contact', 'education','marital','job' ]
    categorical_encoder = OrdinalEncoder()
    df[cols] = categorical_encoder.fit_transform(df[cols])
    #Understand each transformation of categorical values
    encoding = categorical_encoder.categories_
    encoding_feature = lambda x: dict(zip(x, range(len(x))))
    encoding_dict = [encoding_feature(feature_elem) for feature_elem in encoding]
    dict_final = dict(zip(cols, encoding_dict))
    df_preprocessed = df.copy()

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA present in "+ col)
    logger.info('preprocessing banking ok')

    return df_preprocessed, X_columns, y_column, dict_final


def preprocessing_for_orfee(data, conf):
    """
    Preprocessing for the YAR dataset
    Args:
        df: Our Yar dataset
        conf:  conf file

    Returns: Preprocessed YAR Dataset

    """
    #Steps:
    # Drop the NAN rows
    # Deal with the date
    # Select the more relevant columns
    # Do some features engenering
    logger.debug('Cleaning output')

    # Drop the NAN rows
    logger.debug('Drop the NAN rows')
    df = data.dropna()

    # Drop the NAN rows
    logger.debug('Deal with the date')
    df = encode_dates(df, 'date')

    # Select the more relevant columns after EDA:
    logger.debug('Selection of X and Y')
    y_column = u.get_y_column_from_conf(conf)
    continuous_col = ['age', 'exp', 'salaire', 'note']
    category_col = ['cheveux', 'sexe', 'specialite', 'dispo']
    date_col = ['n_days', 'month', 'year']
    X_columns = continuous_col + category_col + ['diplome'] + date_col

    logger.debug('Features Engenering')
    # features engenering dor continuous variables and category variables
    # we use Ordinal Encoder for category variables
    categorical_encoder = OrdinalEncoder()
    df[category_col] = categorical_encoder.fit_transform(df[category_col])
    #Understand each transformation of categorical values
    encoding = categorical_encoder.categories_
    encoding_feature = lambda x: dict(zip(x, range(len(x))))
    encoding_dict = [encoding_feature(feature_elem) for feature_elem in encoding]
    dict_final = dict(zip(category_col, encoding_dict))

    # We use K-bins descretizer for countinuous features
    est = KBinsDiscretizer(n_bins=8, encode='ordinal', strategy='quantile')
    df[continuous_col] = est.fit_transform(df[continuous_col])

    # Special descretizer for diplome as the order important
    dico_diplome = {"bac": 0, "licence": 1, "master": 2, "doctorat": 3}
    df = df.replace({"diplome": dico_diplome})
    dict_final['diplome'] = dico_diplome

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df[X_columns + [y_column]]

    logger.debug('Verification of float and na values ')
    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA present in "+ col)
    logger.debug('preprocessing drift ok')

    return df_preprocessed, X_columns, y_column, dict_final

def preprocessing_for_diabetic(df, conf):
    """
    Preprocessing for the DIABETIC dataset
    Args:
        df: Diabetic dataset
        conf:  conf file

    Returns: Preprocessed Diabetic Dataset

    """
    # Steps:
    # Clean the output (as 0 or 1)
    # one hot elevel, car, zipcode
    # drop id

    logger.debug('Cleaning Output')
    # Cleaning Output:
    df['readmitted'] = df['readmitted'].map({'>30': 1, '<30':1, 'NO': 0})

    logger.debug('mapping Values')
    #Cleaning Other fields:
    for col in  ['race','gender','weight','payer_code','medical_specialty']:
        df[col] = df[col].str.replace('?', 'unknown')


    df['age'] = df['age'].map({'unknown':0, '[0-10)':5, '[10-20)':15,'[20-30)':25,'[30-40)':35,'[40-50)':45,
                           '[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95})

    df['weight'] = df['weight'].map({'unknown':0, '[0-25)':15, '[25-50)':40,'[50-75)':65,'[75-100)':90,'[100-125)':115,
                           '[125-150)':140,'[150-175)':165,'[175-200)':190,'>200':215})

    logger.debug('One hot Encoding')
    # one hot encoding
    cols = ['gender','race','payer_code', 'medical_specialty', 'max_glu_serum','A1Cresult','metformin',
            'repaglinide','nateglinide','chlorpropamide','glimepiride','acetohexamide','glipizide',
            'glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
            'examide','citoglipton','insulin','glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone',
            'metformin-rosiglitazone','metformin-pioglitazone','change','diabetesMed' ]
    categorical_encoder = OrdinalEncoder()
    df[cols] = categorical_encoder.fit_transform(df[cols])
    #Understand each transformation of categorical values
    encoding = categorical_encoder.categories_
    encoding_feature = lambda x: dict(zip(x, range(len(x))))
    encoding_dict = [encoding_feature(feature_elem) for feature_elem in encoding]
    dict_final = dict(zip(cols, encoding_dict))

    logger.debug('Dropping unique columns')
    # Drop id:
    df_preprocessed = df.drop(['encounter_id','patient_nbr'], axis=1)

    logger.debug('Selection of X and Y')
    # returning columns for train test split
    y_column = u.get_y_column_from_conf(conf)
    X_columns = [x for x in df_preprocessed.columns if x != y_column]

    logger.debug('Verification of float and na values')

    # verification:
    for col in df_preprocessed.columns:
        try:
            df_preprocessed[col] = df_preprocessed[col].astype(str).str.replace(',', '.').astype(float)
        except:
            logger.error( col + " cannot be typed as float")
        if df_preprocessed[col].isna().sum() > 0:
            logger.warning( "NA present in "+ col)
    logger.info('preprocessing Diabetic ok')

    return df_preprocessed, X_columns, y_column, dict_final
    

def impute_num(df, cols, cond):
    """
    Replace the NAN in a column by mean by group
    Args:
        df: dataset with NAN
        cols: the columns with the NANs
        cond: variable to group on 

    Returns: dataframe with the nan replaced

    """
    for col in cols:
        df = df.set_index([cond])
        means = df.groupby([cond])[col].mean()
        df[col] = df[col].fillna(means)
        df = df.reset_index()
    return df

def encode_dates(data, var):
    
    # Transform date to dtype format
    df = data.copy()  
    df.loc[:, var] = pd.to_datetime(df[var])
    
    # Encode the date information from the date columns
    df.loc[:, 'month'] = df[var].dt.month
    df.loc[:, 'year'] = df[var].dt.year
    df.loc[:, 'n_days'] = df[var].apply(
        lambda date: (date - df[var].min()).days)
    
    return df.drop(var, axis=1)


def basic_split(df, size_of_test, X_columns, y_column):
    """
    Split the dataframe in train, test sets
    Args:
        df: Dataframe to Split
        size_of_test: proportion of test dataset
        X_columns: Columns for the variables
        y_column: Column for the output

    Returns: Train and test datasets for variables and output

    """
    X_train, X_test, y_train, y_test = train_test_split(df[X_columns], df[y_column],
     test_size = size_of_test)
    return X_train, X_test, y_train, y_test
