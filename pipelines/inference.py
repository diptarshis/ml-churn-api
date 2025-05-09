import pandas as pd
#from sklearn.feature_selection import SelectFromModel
import pickle
import yaml
import joblib
import traceback
import sys

def build_features_from_embeddings(df, categorical_cols, embedding_dir="."):
    df = df.copy()
    for col in categorical_cols:
        embed_path = f"{embedding_dir}/{col}_embeddings.csv"
        embed_df = pd.read_csv(embed_path)
        df = df.merge(embed_df, on=col, how='left')
        df.drop(columns=[col], inplace=True)
    #X = df.drop(columns=[label_col])

    return df

def apply_numeric_preprocessing(df, numeric_cols, stats):
    df = df.copy()

    for col in numeric_cols:
        # Clip outliers using training IQR
        df[col] = df[col].clip(stats[col]['lower'], stats[col]['upper'])

    return df

def transform_with_label_encoders(df, encoders, columns):
    df_encoded = df.copy()
    print("""Inside the transform_with_label_encoders encoders""")
    for col in columns:
        print(col)
        le = encoders[col]
        mapping = {label: idx for idx, label in enumerate(le.classes_)}
        df_encoded[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df_encoded

def transform_ohe_encoder(X, ohe, categorical_cols, feature_names):
    X_ohe = ohe.transform(X[categorical_cols])
    X_ohe_df = pd.DataFrame(X_ohe, columns=feature_names, index=X.index)
    X_dropped = X.drop(columns=categorical_cols)
    X_transformed = pd.concat([X_dropped, X_ohe_df], axis=1)
    return X_transformed

def transform_frequency_encoder(X, categorical_cols, freq_maps):
    X_copy = X.copy()
    for col in categorical_cols:
        if col in X.columns:
            freq_map = freq_maps.get(col, {})
            X_copy[col + '_freq_encode'] = X_copy[col].map(freq_map)
            X_copy[col + '_freq_encode'] = X_copy[col + '_freq_encode'].fillna(0)
    
    X_copy=X_copy.loc[:,[i for i in X_copy.columns if '_freq_encode' in i]]
    return X_copy



def run_inference(PATH_CONFIG, PATH_INPUT_FILE):
    try:
        with open(PATH_CONFIG, "r") as file:
            config = yaml.safe_load(file)

        df = pd.read_csv(PATH_INPUT_FILE)

        if 'Label' in config['label']:
            df = df.loc[:,~df.columns.isin(['Label'])]
            print(df.head())

        # Load a pickle file (e.g., model.pkl or param_dist.pkl)
        with open(config['ARTIFACTS']['ohe_path'], 'rb') as f:
            ohe = pickle.load(f)

        with open(config['ARTIFACTS']['ohe_var_path'], 'rb') as f:
            ohe_vars = pickle.load(f)
        print(ohe_vars)

        with open(config['ARTIFACTS']['feature_names_path'], 'rb') as f:
            feature_names = pickle.load(f)
        print(feature_names)

        with open(config['ARTIFACTS']['freq_var_path'], 'rb') as f:
            freq_vars = pickle.load(f)
        print(freq_vars)

        with open(config['ARTIFACTS']['freq_map_path'], 'rb') as f:
            freq_maps = pickle.load(f)
        print(freq_maps)

        with open(config['ARTIFACTS']['numeric_stats_path'], 'rb') as f:
            numeric_stats = pickle.load(f)
        print(numeric_stats)

        with open(config['ARTIFACTS']['numeric_cols_path'], 'rb') as f:
            numeric_cols = pickle.load(f)
        print(numeric_cols)

        best_model = joblib.load(config['ARTIFACTS']['best_model'])


        knn_imputer_final = joblib.load(config['ARTIFACTS']['imputer_path_knn_final'])

        with open(config['ARTIFACTS']['base_model_feature_select'], 'rb') as f:
            base_model_feature_select = pickle.load(f)

        if config['miss_treatment_numeric']=='knn':
            with open(config['ARTIFACTS']['imputer_path_knn'], 'rb') as f:
                knn_imputer = pickle.load(f)
            print(knn_imputer)
        else:
            with open(config['ARTIFACTS']['imputer_path_bayesian_ridge'], 'rb') as f:
                pmm_imputer = pickle.load(f)
            print(pmm_imputer)

        with open(config['ARTIFACTS']['features_selected_test'], 'rb') as f:
            selected_cols = pickle.load(f)



        X_test_ohe_encoded = transform_ohe_encoder(df.loc[:,ohe_vars], ohe, ohe_vars, feature_names)

        X_test_encoded_freq = transform_frequency_encoder(df, freq_vars, freq_maps) 

        X_test_emb_1 = df.loc[:,freq_vars]
        X_train_ds=build_features_from_embeddings(X_test_emb_1, freq_vars, embedding_dir=config['ARTIFACTS']['path_embeddings'])

        print(X_train_ds.head())


        #####Outlier Treatment########

        test_df_clean = apply_numeric_preprocessing(df.loc[:,numeric_cols], numeric_cols, numeric_stats)

        ##Missing Data Treatment
        if config['miss_treatment_numeric']=='knn':

            test_df_clean_num = pd.DataFrame(
                knn_imputer.transform(test_df_clean[numeric_cols]),
                columns=numeric_cols
            )
        else:
            test_df_clean_num = pd.DataFrame(
            pmm_imputer.transform(test_df_clean[numeric_cols]),
            columns=numeric_cols
            )

        def combine_(X_train_ohe_encoded,X_train_encoded_freq, train_df_clean_num, X_train_ds):
            return pd.concat([X_train_ohe_encoded.reset_index(drop=True),pd.concat([X_train_encoded_freq.reset_index(drop=True),pd.concat([train_df_clean_num.reset_index(drop=True),pd.concat([X_train_ds.reset_index(drop=True),],axis=1)],axis=1).reset_index(drop=True)],axis=1)],axis=1)

        testing_data=combine_(X_test_ohe_encoded,X_test_encoded_freq, test_df_clean, X_train_ds)
        print(testing_data)
        print(testing_data.isnull().sum().sum())

        testing_data = pd.DataFrame(knn_imputer_final.transform(testing_data), columns=testing_data.columns)

        X_val_sel = testing_data.loc[:,selected_cols]

        y_val_prob = best_model.predict_proba(X_val_sel)[:, 1]

        print(y_val_prob)

        df = pd.concat([df, pd.Series(y_val_prob, name="Probability", index=df.index)], axis=1)

        return df

    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        tb = traceback.extract_tb(exc_tb)
        filename, lineno, func, text = tb[-1]
        print(f"Error occurred in inference pipeline: {e} - {filename}, line {lineno}, in {func}")
        raise Exception(f"{e}")
        return pd.DataFrame()





#############################################RUN MAIN#####################################################

# if __name__=="__main__":
#     PATH_CONFIG = "../config.yaml"
#     PATH_INPUT_FILE = "../uploads/inf_data_test.csv"
#     result = run_inference(PATH_CONFIG, PATH_INPUT_FILE)
#     print(result.shape[0], " records in output")