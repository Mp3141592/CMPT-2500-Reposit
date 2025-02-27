#!/usr/bin/env python3

"""
File should process all the data using pandas, split in with sklearn, and then export it with os and shutil.
"""

import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
import warnings
import shutil
import os
import train
import evaluate


warnings.filterwarnings('ignore')


file_path = "/home/machine/cmpt3830/data/raw/CBB_Listings.csv"

df = pd.read_csv(file_path)

#removing duplicates
df.drop_duplicates(inplace= True)


#dropping a column by name
column_to_remove = ['has_leather', 'has_navigation','listing_id','listing_heading', 'listing_type', 'listing_url', 'listing_first_date', 'days_on_market', 'dealer_id', 'dealer_name', 'dealer_street', 'dealer_city', 'dealer_province', 'dealer_postal_code', 'dealer_url', 'dealer_email', 'dealer_phone', 'dealer_type', 'vehicle_id', 'uvc', 'price_analysis', 'price_history_delimited', 'distance_to_dealer', 'location_score', 'listing_dropoff_date', 'certified']
df.drop(columns=column_to_remove, inplace=True)


# Outlier detection using IQR for each numerical column
for col in df.select_dtypes(include='number').columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Define the outlier range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    #print(f'Outliers in {col}: \n{outliers}\n')

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter the dataframe
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)] # removed the extra indent
    return filtered_df

# Applying outlier removal for 'number_price_changes' and 'wheelbase_from_vin'
df_cleaned = remove_outliers_iqr(df, 'number_price_changes')
df_cleaned = remove_outliers_iqr(df, 'wheelbase_from_vin')

# Display cleaned DataFrame
#df_cleaned.head()

df = df_cleaned


numerical_columns = ['mileage', 'price', 'msrp', 'model_year',
                    'wheelbase_from_vin', 'number_price_changes']

categorical_columns = ['stock_type', 'vin', 'make', 'model', 'series', 'style',
                       'exterior_color', 'exterior_color_category', 'interior_color',
                       'interior_color_category', 'drivetrain_from_vin', 'engine_from_vin',
                       'transmission_from_vin', 'fuel_type_from_vin']


# Columns where 0 is invalid and should be treated as missing values (NaN)
invalid_zero_columns = ['mileage', 'price', 'msrp', 'wheelbase_from_vin', 'number_price_changes']

# Replacing 0 with NaN in these columns
df[invalid_zero_columns] = df[invalid_zero_columns].replace(0, np.nan)

# For numerical columns, fill missing values with median
for col in invalid_zero_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Check if 0 values have been handled correctly
#print(df[invalid_zero_columns].isnull().sum())


for col in numerical_columns:
    df[col].fillna(df[col].mean(), inplace=True)


for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

for i in categorical_columns:
  df = df.astype({i: 'category'})


# Replacing 6 with M and 7 with A
df = df.replace({"transmission_from_vin": "6"}, {"transmission_from_vin": "M"})
df = df.replace({"transmission_from_vin": "7"}, {"transmission_from_vin": "A"})


# Replaces values less than 1000 with the mean for price and msrp
df['price'] = df['price'].mask(df['price'] < 1000, df['price'].mean())
df['msrp'] = df['msrp'].mask(df['msrp'] < 1000, df['msrp'].mean())


# Replaces all mileage values with less than 1000 and a stock type of USED with the mean value (Note: | is or, NOT AND!)
df['mileage'] = df['mileage'].mask((df['mileage'] < 1000) & (df['stock_type'] == 'USED'), df['mileage'].mean(), inplace=False)


# Categories with very few of a value
less = ['exterior_color_category', 'interior_color_category', 'make', 'fuel_type_from_vin', 'model']


# Removing those values
for cat in less:
  value_counts = df[f"{cat}"].value_counts()
  df = df[df[cat].isin(value_counts[value_counts > 3].index)]


# Dropping again 
df = df.drop(columns=["exterior_color", "interior_color", 'vin','exterior_color_category',	'interior_color_category',	
                      'wheelbase_from_vin',	'drivetrain_from_vin',	'engine_from_vin','number_price_changes'])


# Split to X and y
X = df[['make','mileage','model_year','transmission_from_vin','stock_type','msrp']]
y = df['price']


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)


# Put into csvs
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)


# Put names of csvs into list
pro_csv = ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv"]


# Folder to move processed data into
target_folder = '/home/machine/cmpt3830/data/processed'  


# For loop to process data
for proc in pro_csv:

   # Find path for target folder 
   destination_path = os.path.join(target_folder, proc)
   
   # Check if file already made and avoids making another
   if os.path.exists(destination_path):
    print(f"The file '{destination_path}' already exists.") 
   
   # Moves file to folder if first instance
   else: 
    shutil.move(proc, destination_path)

# Temp module running until main py made
X_train_path = "/home/machine/cmpt3830/data/processed/X_train.csv"
X_test_path = "/home/machine/cmpt3830/data/processed/X_test.csv"
y_train_path = "/home/machine/cmpt3830/data/processed/y_train.csv"
y_test_path = "/home/machine/cmpt3830/data/processed/y_test.csv"

from train import Train
from evaluate import Eval

print("Training Begins")
training = Train(X_train_path, X_test_path, y_train_path, y_test_path)

h = training.trainmodel()

print("Training Finishes")

print('Evaluation Begins')
model_path = '/home/machine/cmpt3830/models/ridge_model.jlib'

eval = Eval(y_test_path, X_test_path, model_path, h)

eval.Evalulate()

print("Evaluation Finishes")

print("Should print this sentence if code is run to end.")