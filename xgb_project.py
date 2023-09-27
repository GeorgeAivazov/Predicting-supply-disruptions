import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import optuna
from optuna.samplers import TPESampler



##ПОДГОТОВКА ДАННЫХ
#Скачивание данных
df     = pd.read_csv('data/train_AIC.csv')
X_test = pd.read_csv('data/test_AIC.csv')


#Заполняем пропуски
features      = df.    columns.values.tolist()
features_test = X_test.columns.values.tolist()
df    [features     ] = df    [features     ].fillna(df[features ].mode())
X_test[features_test] = X_test[features_test].fillna(df[features_test].mode())


#Уменьшаем количество индексов
values = df['Поставщик'].value_counts().index[0:1000]
mask_df   = df    ['Поставщик'].isin(values)
mask_test = X_test['Поставщик'].isin(values)
df.loc    [~mask_df,   'Поставщик'] = 0
X_test.loc[~mask_test, 'Поставщик'] = 0


values = df['Материал'].value_counts().index[0:1000]
mask_df   = df    ['Материал'].isin(values)
mask_test = X_test['Материал'].isin(values)
df.loc    [~mask_df,   'Материал'] = 0
X_test.loc[~mask_test, 'Материал'] = 0


#Определяем категориальные признаки
categorical_features = ['Материал','Поставщик',"Вариант поставки","День недели 2","Месяц1","Месяц2","Месяц3","НРП","Согласование заказа 1","Завод","ЕИ","Закупочная организация","Согласование заказа 2","Согласование заказа 3"]

df[categorical_features] = df[categorical_features].astype('int32').astype("category")

for i in categorical_features:
     X_test[i] = X_test[i].astype('int32').astype("category").cat.set_categories(df[i].cat.categories)
     X_test[i] = X_test[i].cat.add_categories('NaN').fillna('NaN')


#Разделяем датасет на тренировочный  валидационный
X,y = (df.loc[:, df.columns !='y'], df.loc[:,"y"])
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.05, random_state=42)


# Сэмплируем тренировочный датасет
ran=RandomOverSampler(random_state=42)
X_train, y_train= ran.fit_resample(X_train,y_train)


#Удаляем выбросы
index = X_train.keys()
for i in index:
    if i in categorical_features: continue
    X_train = X_train[X_train[i] >= X_train[i].quantile(0.000002)]
    X_train = X_train[X_train[i] <= X_train[i].quantile(0.999998)]

y_train = y_train[X_train.index.values[:]]



##НАСТРОЙКА ГИПЕРПАРАМЕТРОВ

def objective_xgboost(trial):
    """Определяем целевую функцию"""

    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 11),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.04,log=True),
        'n_estimators': trial.suggest_int('n_estimators', 5000, 9000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.9, 1.0,log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0,log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0,log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.,log=True),
        'objective':'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state':42,
        'seed':42,
        'enable_categorical': True,
        'verbosity':0
    }

    # Тренируем модель
    optuna_model = XGBClassifier(**params)
    optuna_model.fit(X_train, y_train)

    # Делаем предсказание
    y_pred = optuna_model.predict(X_valid)

    # Вычисляем метрику f1_score на валидационном датасете
    f1_sc = f1_score(y_valid, y_pred)
    return f1_sc


#Процесс настройки гиперпараметров
sampler = TPESampler(seed=42)
study_xgboost = optuna.create_study(sampler=sampler, direction='maximize')
study_xgboost.optimize(objective_xgboost, n_trials=120)


#Вывод в командную строку лучших гиперпараметров
trial_xgboost = study_xgboost.best_trial
print('Number of finished trials: {}'.format(len(study_xgboost.trials)))
print('Best trial:')
print('  Value: {}'.format(trial_xgboost.value))
print('  Params: ')
for key, value in trial_xgboost.params.items():
    print('    {}: {}'.format(key, value))


#Сохранение в переменную(словарь) лучших гиперпараметров
best_xgboost_parameters = study_xgboost.best_params



##ОБУЧЕНИЕ НА ПОЛНОМ НАБОРЕ ДАННЫХ 
# Сэмплируем полный датасет
ran=RandomOverSampler(random_state=42)
X, y= ran.fit_resample(X,y)


#Удаляем выбросы
index = X.keys()
for i in index:
    if i in categorical_features: continue
    X = X[X[i] >= X[i].quantile(0.000002)]
    X = X[X[i] <= X[i].quantile(0.999998)]

y = y[X.index.values[:]]


#Задаем гиперпараметры
params = {
        'max_depth': best_xgboost_parameters['max_depth'],
        'learning_rate': best_xgboost_parameters['learning_rate'],
        'n_estimators': best_xgboost_parameters['n_estimators'],
        'min_child_weight': best_xgboost_parameters['min_child_weight'],
        'subsample': best_xgboost_parameters['subsample'],
        'colsample_bytree': best_xgboost_parameters['colsample_bytree'],
        'reg_alpha': best_xgboost_parameters['reg_alpha'],
        'reg_lambda': best_xgboost_parameters['reg_lambda'],
        'objective':'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state':42,
        'seed':42,
        'enable_categorical': True,
        'verbosity':0
        }


#Инициализируем и тренируем модель
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(X, y)



##ПРЕДСКАЗАНИЕ НА ТЕСТОВОМ НАБОРЕ ДАННЫХ
pred = xgb_model.predict(X_test)
pred = pd.DataFrame(pred,dtype='int32')
pred.index.names = ['id']
pred =pred.set_axis(['value'], axis=1)
pred.to_csv('submission.csv')