import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV, SelectFromModel, VarianceThreshold
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
import yaml
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
import traceback
import sys


def load_and_merge_data(features_path, labels_path):
    df = pd.read_csv(features_path)
    df_label = pd.read_csv(labels_path)
    return pd.concat([df, df_label], axis=1)

def filter_columns_by_fill_rate(df, threshold=70):
    fill_rates = (df.notnull().sum() / len(df)) * 100
    selected_columns = fill_rates[fill_rates > threshold].index.tolist()
    return df[selected_columns], fill_rates


def generate_ohe_freq_vars_for_emb(X_train,dep, threshold):
    """
    Generate the Feature-list  One Hot Encoding Features for encoding the data
    """
    uniques_ = {i:X_train[i].nunique() for i in X_train.select_dtypes(include=['object']).columns}
    ohe_vars = [i for i,j in uniques_.items() if j < threshold if i not in dep]
    print(ohe_vars)
    freq_vars = [i for i,j in uniques_.items() if j >= threshold if i not in dep]
    print(freq_vars)

    return ohe_vars, freq_vars

def filter_categorical_columns(df):
    categorical_fields = df.select_dtypes(include=['object']).columns.tolist()
    return categorical_fields


def train_test_split_(df_combined, test_size,path_to_write_train,path_to_write_test):
    X_train, X_test = train_test_split(df_combined, test_size=test_size, random_state=42)
    X_train.to_csv(path_to_write_train,index=False)
    X_test.to_csv(path_to_write_test,index=False)
    return X_train, X_test

def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def fit_ohe_encoder(X_train, categorical_cols,ohe_path,ohe_var_path, feature_names_path):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(X_train[categorical_cols])
    feature_names = ohe.get_feature_names_out(categorical_cols)
    save_obj(ohe,ohe_path)
    save_obj(categorical_cols,ohe_var_path)
    save_obj(feature_names,feature_names_path)
    return ohe, feature_names

def transform_ohe_encoder(X, ohe, categorical_cols, feature_names):
    X_ohe = ohe.transform(X[categorical_cols])
    X_ohe_df = pd.DataFrame(X_ohe, columns=feature_names, index=X.index)
    X_dropped = X.drop(columns=categorical_cols)
    X_transformed = pd.concat([X_dropped, X_ohe_df], axis=1)
    return X_transformed


# --- Fit Frequency Encoder for multiple columns ---
def fit_frequency_encoder(X_train, categorical_cols,freq_map_path,freq_var_path):
    freq_maps = {}
    for col in categorical_cols:
        freq_map = X_train[col].value_counts().to_dict()
        freq_maps[col] = freq_map
    save_obj(freq_maps,freq_map_path)
    save_obj(categorical_cols,freq_var_path)
    return freq_maps

# --- Transform Data using Fitted Frequency Encoder ---
def transform_frequency_encoder(X, categorical_cols, freq_maps):
    X_copy = X.copy()
    for col in categorical_cols:
        if col in X.columns:
            freq_map = freq_maps.get(col, {})
            X_copy[col + '_freq_encode'] = X_copy[col].map(freq_map)
            X_copy[col + '_freq_encode'] = X_copy[col + '_freq_encode'].fillna(0)
    
    X_copy=X_copy.loc[:,[i for i in X_copy.columns if '_freq_encode' in i]]
    return X_copy


def fit_label_encoders(df, columns, config):
    encoders = {}
    print("""Inside the fitting label encoders""")
    for col in columns:
        print(col)
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    ###Save Label Encoders
    save_obj(encoders,config['ARTIFACTS']['encoder_path'])
    return encoders

def transform_with_label_encoders(df, encoders, columns):
    df_encoded = df.copy()
    print("""Inside the transform_with_label_encoders encoders""")
    for col in columns:
        print(col)
        le = encoders[col]
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        df_encoded[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df_encoded



# #####Neural Net Embedding
def neural_net_embedding(X_train, X_test, freq_vars,label_col,path_embeddings,hidden_embed = 50, epochs=3, batch_size = 32):
    """
            Neural Net Embedding for variables with high cardinality
    """

    try:
        encoders = fit_label_encoders(X_train, freq_vars)
        #print(encoders)
        train_encoded = transform_with_label_encoders(X_train, encoders, freq_vars)
        #print(type(train_encoded))
        test_encoded = transform_with_label_encoders(X_test, encoders, freq_vars)
        #print(type(test_encoded))


        X_train_emb = [train_encoded[col].values for col in freq_vars]
        X_val_emb = [test_encoded[col].values for col in freq_vars]
        y_train_emb = train_encoded[label_col].replace({-1: 0}).values
        y_val_emb = test_encoded[label_col].replace({-1: 0}).values

        # --- Build the embedding model ---
        embedding_inputs = []
        embedding_outputs = []

        for col in freq_vars:
            n_unique = train_encoded[col].nunique()
            embed_dim = min(hidden_embed, (n_unique + 1) // 2)

            input_layer = Input(shape=(1,), name=f"{col}_input")
            embedding_layer = Embedding(input_dim=n_unique + 1, output_dim=embed_dim, name=f"{col}_embed")(input_layer)
            flatten_layer = Flatten()(embedding_layer)

            embedding_inputs.append(input_layer)
            embedding_outputs.append(flatten_layer)

        # --- Combine and define output ---
        x = Concatenate()(embedding_outputs)
        x = Dense(64, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=embedding_inputs, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])

        # --- Train model ---
        model.fit(X_train_emb, y_train_emb, epochs=epochs, batch_size=batch_size, validation_data=(X_val_emb, y_val_emb))
        save_obj(model,f"{path_embeddings}embedding_model.pkl")
    # --- Extract and save embeddings ---
        for col in freq_vars:
            print(col)
            layer = model.get_layer(f"{col}_embed")
            weights = layer.get_weights()[0]
            le = encoders[col]
            seen_indices = np.arange(len(le.classes_))  # Only valid indices
            values = le.inverse_transform(seen_indices)

            # Slice weights and only assign decoded values to matching rows
            embed_df = pd.DataFrame(
                weights[:len(values)],
                columns=[f"{col}_embed_{i}" for i in range(weights.shape[1])]
            )
            embed_df[col] = values  # This will now match in length 
            embed_df.to_csv(f"{path_embeddings}{col}_embeddings.csv", index=False)
            print(f"Saved embeddings for {col} → {col}_embeddings.csv")
            
        return True
    except:
        return False  

def build_features_from_embeddings(df, label_col, categorical_cols, embedding_dir="."):
    df = df.copy()
    for col in categorical_cols:
        embed_path = f"{embedding_dir}/{col}_embeddings.csv"
        embed_df = pd.read_csv(embed_path)
        df = df.merge(embed_df, on=col, how='left')
        df.drop(columns=[col], inplace=True)
    X = df.drop(columns=[label_col])

    if label_col in df.columns:
        y = df[label_col]
    else:
        y=pd.DataFrame()


    return X, y

def fit_numeric_preprocessing(df, numeric_cols):
    """Outlier Treatment: Extract Stats from a group of features"""
    stats = {}

    for col in numeric_cols:
        col_data = df[col].dropna()
        median = col_data.median()
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        stats[col] = {
            "median": median,
            "lower": lower,
            "upper": upper
        }

    return stats

def apply_numeric_preprocessing(df, numeric_cols, stats):
    df = df.copy()

    for col in numeric_cols:
        # Clip outliers using training IQR
        df[col] = df[col].clip(stats[col]['lower'], stats[col]['upper'])

    return df


def outlier_treatment(X_train,X_test,config):
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    ser = X_train.loc[:,numeric_cols].isnull().sum()/X_train.shape[0]<(100-config['fill_rate_cutoff'])/100
    numeric_cols=ser[ser].index.tolist()
    numeric_cols = [i for i in numeric_cols if config['label'] not in i]
    save_obj(numeric_cols, config['ARTIFACTS']['numeric_cols_path'])

    # Fit preprocessing parameters on train set
    numeric_stats = fit_numeric_preprocessing(X_train, numeric_cols)
    save_obj(numeric_stats, config['ARTIFACTS']['numeric_stats_path'])

    # Apply on train and test set using same stats
    train_df_clean = apply_numeric_preprocessing(X_train, numeric_cols, numeric_stats)
    test_df_clean = apply_numeric_preprocessing(X_test, numeric_cols, numeric_stats)

    train_df_clean_num = train_df_clean.loc[:,numeric_cols]
    test_df_clean_num = test_df_clean.loc[:,numeric_cols]

    return train_df_clean_num, test_df_clean_num, numeric_cols

def missing_treatment(train_df_clean_num, test_df_clean_num,numeric_cols,method,config):
    """Two types of missing value treatments, KNN takes less time """

    print("In missing value treatment")
    if method =="knn":
    # Use BayesianRidge or DecisionTreeRegressor as estimator
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
        # Fit on train, transform on both
        train_df_clean_num = pd.DataFrame(
            knn_imputer.fit_transform(train_df_clean_num[numeric_cols]),
            columns=numeric_cols
        )

        save_obj(knn_imputer,config['ARTIFACTS']['imputer_path_knn'])

        ##Write knn_imputer to Artifacts
        test_df_clean_num = pd.DataFrame(
            knn_imputer.transform(test_df_clean_num[numeric_cols]),
            columns=numeric_cols
        )
    else:
        pmm_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
                # Fit on train, transform on both
        train_df_clean_num = pd.DataFrame(
            pmm_imputer.fit_transform(train_df_clean_num[numeric_cols]),
            columns=numeric_cols
        )
        save_obj(pmm_imputer,config['ARTIFACTS']['imputer_path_bayesian_ridge'])

        ##Write pmm_imputer to Artifacts
        test_df_clean_num = pd.DataFrame(
            pmm_imputer.transform(test_df_clean_num[numeric_cols]),
            columns=numeric_cols
        )
    return train_df_clean_num, test_df_clean_num


def combine_(X_train_ohe_encoded,X_train_encoded_freq, train_df_clean_num, X_train_ds, y_train_ds):
    return pd.concat([X_train_ohe_encoded.reset_index(drop=True),pd.concat([X_train_encoded_freq.reset_index(drop=True),pd.concat([train_df_clean_num.reset_index(drop=True),pd.concat([X_train_ds.reset_index(drop=True),y_train_ds.reset_index(drop=True)],axis=1)],axis=1).reset_index(drop=True)],axis=1)],axis=1)

def replace_0_1(df):
    df['Label'] = df['Label'].replace({-1: 0})
    return df

def main_feature_generation(PATH_CONFIG = "../config.yaml"):
    try:

        # Load YAML config
        with open(PATH_CONFIG, "r") as file:
            config = yaml.safe_load(file)

        FEATURES_PATH = config['PATHS']['features_path']
        LABELS_PATH = config['PATHS']['labels_path']
        ARTIFACTS_PATH = config['ARTIFACTS']['path']
        TRAIN_PATH = config['ARTIFACTS']['train_path']
        TEST_PATH = config['ARTIFACTS']['test_path']

        fill_rate_cutoff=config['fill_rate_cutoff']
        test_size = config['test_size']
        threshold_ohe = config['threshold_ohe']
        label_name=config['label']


        ohe_path=config['ARTIFACTS']['ohe_path']
        ohe_var_path=config['ARTIFACTS']['ohe_var_path']
        feature_names_path=config['ARTIFACTS']['feature_names_path']
        freq_map_path = config['ARTIFACTS']['freq_map_path']
        freq_var_path = config['ARTIFACTS']['freq_var_path']

        os.makedirs(ARTIFACTS_PATH, exist_ok=True)

        df_combined=load_and_merge_data(features_path = FEATURES_PATH,labels_path = LABELS_PATH)
        df_combined,fill_rates=filter_columns_by_fill_rate(df_combined,threshold = fill_rate_cutoff)


        X_train, X_test=train_test_split_(df_combined, test_size=test_size, path_to_write_train = TRAIN_PATH, path_to_write_test = TEST_PATH)

        ###One Hot Encoding
        ohe_vars, freq_vars=generate_ohe_freq_vars_for_emb(X_train,dep=label_name, threshold=threshold_ohe)


        ohe, feature_names = fit_ohe_encoder(X_train, ohe_vars,ohe_path,ohe_var_path, feature_names_path)
        X_train_ohe_encoded = transform_ohe_encoder(X_train.loc[:,ohe_vars], ohe, ohe_vars, feature_names)
        X_test_ohe_encoded = transform_ohe_encoder(X_test.loc[:,ohe_vars], ohe, ohe_vars, feature_names)


        ####Frequency Encoder
        freq_maps = fit_frequency_encoder(X_train, freq_vars, freq_map_path,freq_var_path)
        X_train_encoded_freq =transform_frequency_encoder(X_train, freq_vars, freq_maps)
        X_test_encoded_freq = transform_frequency_encoder(X_test, freq_vars, freq_maps) 


        #####Neural Net Embedding
        neural_net_embed=neural_net_embedding(X_train, X_test, freq_vars,
                                            label_col = config['label'],
                                            path_embeddings = config['ARTIFACTS']['path_embeddings'], 
                                            hidden_embed = config['hidden_embed'], epochs=config['epochs'], 
                                            batch_size = config['batch_size'])
        print(neural_net_embed)
        ####
        X_train_emb_1 = pd.concat([X_train.loc[:,freq_vars],X_train.loc[:,[config['label']]]],axis=1)
        X_test_emb_1 = pd.concat([X_test.loc[:,freq_vars],X_test.loc[:,[config['label']]]],axis=1)
        X_train_ds, y_train_ds=build_features_from_embeddings(X_train_emb_1, label_col = config['label'], categorical_cols = freq_vars, embedding_dir=config['ARTIFACTS']['path_embeddings'])
        X_test_ds, y_test_ds=build_features_from_embeddings(X_test_emb_1, label_col = config['label'], categorical_cols = freq_vars, embedding_dir=config['ARTIFACTS']['path_embeddings'])

        print(True)

        ################################FOR OUTLIER TREATMENT############################


        train_df_clean_num, test_df_clean_num,numeric_cols  = outlier_treatment(X_train,X_test,config)
        print(True)
        ##############################FOR MISSING VALUE TREATMENT######################


        train_df_clean_num, test_df_clean_num = missing_treatment(train_df_clean_num, test_df_clean_num, numeric_cols, method=config['miss_treatment_numeric'],config=config)
        print(True)


        training_data=combine_(X_train_ohe_encoded,X_train_encoded_freq, train_df_clean_num, X_train_ds, y_train_ds)
        testing_data=combine_(X_test_ohe_encoded,X_test_encoded_freq, test_df_clean_num, X_test_ds, y_test_ds)
        print(True)
        training_data = replace_0_1(training_data)
        testing_data = replace_0_1(testing_data)
        print(True)
        #knn_imputer = KNNImputer(n_neighbors=config['knn_final_imputation'], weights='uniform')
        #estimator = ExtraTreesRegressor(n_estimators=10, random_state=0, n_jobs=-1)
        #imputer = IterativeImputer(estimator=estimator, max_iter=5, random_state=0, verbose=0)
        #imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0, n_nearest_features=15, initial_strategy="mean")
        #imputer = IterativeImputer(random_state=100, max_iter=2)
        imputer = SimpleImputer(strategy='median')


        training_data_impute = training_data.loc[:,~training_data.columns.isin([label_name])]
        testing_data_impute = testing_data.loc[:,~testing_data.columns.isin([label_name])]
        
        testing_data_final=testing_data.copy()
        training_data_final=training_data.copy()

        if training_data.isnull().sum().sum()>0 or testing_data.isnull().sum().sum()>0:
            print("Missing data observed in merged data")
            training_data_impute = pd.DataFrame(imputer.fit_transform(training_data_impute), columns=training_data_impute.columns)
            training_data_final = pd.concat([training_data_impute,training_data[[label_name]]],axis=1)
            #save_obj(knn_imputer,config['ARTIFACTS']['imputer_path_knn_final'])
            joblib.dump(imputer,config['ARTIFACTS']['imputer_path_knn_final'])
            
            testing_data_impute = pd.DataFrame(imputer.transform(testing_data_impute), columns=testing_data_impute.columns)
            testing_data_final = pd.concat([testing_data_impute,testing_data[[label_name]]],axis=1)

        print(True)

        training_data_final.to_csv(f'{config['modelling_data_path']}training_data.csv',index=False)
        testing_data_final.to_csv(f'{config['modelling_data_path']}testing_data.csv',index=False)
        
        return True
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)
        filename, lineno, func, text = tb[-1]
        
        print(f"Error occurred in feature_generation pipeline: {e} - {filename}, line {lineno}, in {func}")
        return False


#############################################RUN MAIN#####################################################

if __name__=="__main__":
    result=main_feature_generation(PATH_CONFIG = "../config.yaml")
    print(result, "output")