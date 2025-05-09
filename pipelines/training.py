from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import RFECV,SelectFromModel
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform, randint
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform
import numpy as np
import yaml

import numpy as np
import pickle
import joblib
import traceback
import sys


def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def modelling_data_read(config):
    train=pd.read_csv(f'{config['modelling_data_path']}training_data.csv',low_memory=True, dtype=np.float32)
    test=pd.read_csv(f'{config['modelling_data_path']}testing_data.csv',low_memory=True, dtype=np.float32)
    dep = [config['label']]
    X_train = train.loc[:,~train.columns.isin(dep)]
    y_train = train.loc[:,train.columns.isin(dep)]
    X_val = test.loc[:,~test.columns.isin(dep)]
    y_val = test.loc[:,test.columns.isin(dep)]
    return X_train, y_train, X_val, y_val

def feature_selection(X_train,y_train, X_val,config):
    # Fit base model on all features
    base_model = RandomForestClassifier(n_estimators=config['FEATURE_SELECT_PARAMS']['n_estimators'], 
                                        random_state=42,n_jobs=-1)
    base_model.fit(X_train, y_train.values.ravel())
    ###base model####write
    save_obj(base_model, config['ARTIFACTS']['base_model_feature_select'])
    # Select important features only
    selector = SelectFromModel(base_model, threshold="median", prefit=True)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    #selected_cols=X_val_sel.columns
    selected_cols = X_train.columns[selector.get_support()]
    save_obj(selected_cols,config['ARTIFACTS']['features_selected_test'])

    return X_train_sel, X_val_sel

def hyperparameter_tuning(config,X_train_sel,y_train,X_val_sel,y_val):

    # Build param_dist dictionary
    param_dist = {}
    for param, spec in config['PARAM_DIST'].items():
        if 'distribution' in spec:
            if spec['distribution'] == 'randint':
                param_dist[param] = randint(*spec['range'])
            else:
                raise NotImplementedError(f"Distribution {spec['distribution']} not supported.")
        elif 'choices' in spec:
            param_dist[param] = [None if x == 'null' else x for x in spec['choices']]
        else:
            raise ValueError(f"Unknown spec format for {param}: {spec}")


    param_list = list(ParameterSampler(param_dist, n_iter=2, random_state=42))


    # Step 4: Hyperparameter tuning loop
    best_score = -1
    best_model = None
    best_params = None
    results = []

    for i, params in enumerate(param_list):
        print(params)
        model = RandomForestClassifier(random_state=42, **params, n_jobs=-1)
        model.fit(X_train_sel, y_train.values.ravel())

        # Predict probabilities and compute AUC
        y_val_prob = model.predict_proba(X_val_sel)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_prob)
        print(val_auc)
        results.append({'params': params, 'val_auc': val_auc})

        if val_auc > best_score:
            best_score = val_auc
            best_model = model
            best_params = params

    print(best_score)
    print(best_params)

    joblib.dump(best_model, config['ARTIFACTS']['best_model'])

    return best_model, best_score, best_params

def ks(data=None,target=None, prob=None,best_score=None):
    try:
        data['target0'] = 1 - data[target]
        data['bucket'] = pd.qcut(data[prob], 10)
        grouped = data.groupby('bucket', as_index = False)
        kstable = pd.DataFrame()
        kstable['min_prob'] = grouped.min()[prob]
        kstable['max_prob'] = grouped.max()[prob]
        kstable['events']   = grouped.sum()[target]
        kstable['nonevents'] = grouped.sum()['target0']
        kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
        kstable['event_rate'] = (kstable.events / data[target].sum()).apply('{0:.2%}'.format)
        kstable['nonevent_rate'] = (kstable.nonevents / data['target0'].sum()).apply('{0:.2%}'.format)
        kstable['cum_eventrate']=(kstable.events / data[target].sum()).cumsum()
        kstable['cum_noneventrate']=(kstable.nonevents / data['target0'].sum()).cumsum()
        kstable['KS'] = np.round(kstable['cum_eventrate']-kstable['cum_noneventrate'], 3) * 100

        #Formating
        kstable['cum_eventrate']= kstable['cum_eventrate'].apply('{0:.2%}'.format)
        kstable['cum_noneventrate']= kstable['cum_noneventrate'].apply('{0:.2%}'.format)
        kstable['Validation_Best_Score'] = best_score
        kstable.index = range(1,11)
        kstable.index.rename('Decile', inplace=True)
        pd.set_option('display.max_columns', 9)
        print(kstable)
        
        #Display KS
        from colorama import Fore
        print(Fore.WHITE + "KS is " + str(max(kstable['KS']))+"%"+ " at decile " + str((kstable.index[kstable['KS']==max(kstable['KS'])][0])))
    except:
        kstable = pd.DataFrame()
    
    return(kstable)


def model_validation_statistics(config,best_model, X_train_sel, y_train,X_val_sel, y_val,best_score):
    y_train_prob = best_model.predict_proba(X_train_sel)[:, 1]
    y_val_prob = best_model.predict_proba(X_val_sel)[:, 1]
    data_ap_train = pd.DataFrame({'Actual':y_train.values.ravel(),'Predicted':y_train_prob})
    data_ap_val = pd.DataFrame({'Actual':y_val.values.ravel(),'Predicted':y_val_prob})
    ks_train=ks(data=data_ap_train,target='Actual',prob='Predicted')
    ks_train.to_csv(f'{config['ARTIFACTS']['model_validation_stats']}ks_train.csv',index=False)

    ks_val=ks(data=data_ap_val,target='Actual',prob='Predicted',best_score=best_score)
    ks_val.to_csv(f'{config['ARTIFACTS']['model_validation_stats']}ks_val.csv',index=False)

    return ks_train, ks_val


def main_model_training(PATH_CONFIG):

    try:

        with open(PATH_CONFIG, "r") as file:
            config = yaml.safe_load(file)

        X_train, y_train, X_val, y_val=modelling_data_read(config)
        print(X_train.shape)

        X_train_sel, X_val_sel = feature_selection(X_train,y_train, X_val,config)

        best_model, best_score, best_params=hyperparameter_tuning(config,X_train_sel,y_train,X_val_sel,y_val)

        ks_train, ks_val = model_validation_statistics(config,best_model, X_train_sel, y_train,X_val_sel, y_val,best_score)

        print(ks_train)
        print(ks_val)

        return True

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)
        filename, lineno, func, text = tb[-1]
        print(f"Error occurred in training pipeline: {e} - {filename}, line {lineno}, in {func}")
        return False

if __name__=="__main__":
    result=main_model_training(PATH_CONFIG = "../config.yaml")
    print(result, "output")