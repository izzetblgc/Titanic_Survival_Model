import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#Veri Ön işleme için kullanacağımız fonksiyonlar
from helpers.eda import *
from helpers.data_prep import *


def load_titanic():
    return pd.read_csv("datasets/titanic.csv")


def titanic_data_prep():
    dataframe = load_titanic()

    dataframe.columns = [col.upper() for col in dataframe.columns]
    # Cabin bool
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
    # Name count
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
    # name word count
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
    # name dr
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))
    # name title
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
    # family size
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
    # is alone
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

    # Eksik Değerler

    # titleların medianına göre eksik değerlerin doldurulması
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))

    # age_pclass
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]

    # age level
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

    # sex x age
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
                (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
                (dataframe['AGE'] > 21) & (dataframe['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    # Aykırı Değerler
    for col in num_cols:
        print(col, check_outlier(dataframe, col))
    for col in num_cols:
        replace_with_thresholds(dataframe, col)

    # Eksik Değerler
    missing_values_table(dataframe)
    dataframe.drop(["CABIN", "TICKET", "NAME"], inplace=True, axis=1)

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,
                                axis=0)

    # Label Encoding

    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # Rare Encoding

    dataframe = rare_encoder(dataframe, 0.01)
    rare_analyser(dataframe, "SURVIVED", cat_cols)

    # One Hot Encoding

    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]

    dataframe = one_hot_encoder(dataframe, ohe_cols)

    #Verimizin yeni halini yeniden gözlemleyelim

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    #Verideki nunique oranları 0.01den düşük verilerin verisetinden çıkarılması
    useless_columns = useless_cols(dataframe)

    dataframe.drop(useless_columns, axis=1, inplace=True)

    #Numerik değerlerin standartlaştırılması

    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

    return dataframe


df = titanic_data_prep()

# Modelin kurulması

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# MODEL
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)  # 0.8171641791044776


