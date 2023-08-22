from joblib import load

from flask import Flask, jsonify, request, jsonify, render_template
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from imblearn.under_sampling  import RandomUnderSampler

from lightgbm import LGBMClassifier


# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
app = Flask(__name__)


# On charge les données
#data=pd.read_csv("https://www.dropbox.com/scl/fi/e9fsy8ldmvxnhsvh13kr7/merged_data_feat_eng.csv?#rlkey=85cddvaz2zizz497mtbl56u9j&dl=1")
#app_train_csv=pd.read_csv("https://www.dropbox.com/scl/fi/tanyjwas96srv4ymcs7vs/application_train#.csv?rlkey=0rcplw3g21h238rj7bm82t1bk&dl=1")
#app_train=app_train_csv.copy()
#app_test_csv = #pd.read_csv("https://www.dropbox.com/scl/fi/a43k3r3yppfzhv80lrl1s/application_test.csv?#rlkey=x8sun1p22yop0qnsm7qe2knbz&dl=1")
#app_test=app_test_csv.copy()

#data_train = data[data['SK_ID_CURR'].isin(app_train.SK_ID_CURR)]
#data_test = data[data['SK_ID_CURR'].isin(app_test.SK_ID_CURR)]

#data_test = data_test.drop('TARGET', axis=1)

#data_train.set_index('SK_ID_CURR', inplace=True)
#data_test.set_index('SK_ID_CURR', inplace=True)

data_train=pd.read_csv("app_train.csv")
data_test=pd.read_csv("app_test.csv")

#Feature engineering fonction 

def features_engineering(data_train, data_test):

    # Cette fonction regroupe toutes les opérations de features engineering
    # mises en place sur les sets train & test

    #############################################
    # LABEL ENCODING
    #############################################
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in data_train:
        if data_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(data_train[col])
                # Transform both training and testing data
                data_train[col] = le.transform(data_train[col])
                data_test[col] = le.transform(data_test[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1

    ############################################
    # ONE HOT ENCODING
    ############################################
    # one-hot encoding of categorical variables
    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)

    train_labels = data_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    data_train, data_test = data_train.align(data_test, join = 'inner', axis = 1)
    # Add the target back in
    data_train['TARGET'] = train_labels

    return data_train, data_test




# Preprocessing function
def preprocesseur(df_train, df_test):
    
    # Cette fonction permet d'imputer les valeurs manquantes dans
    # chaque dataset et aussi d'appliquer un MinMaxScaler

    # Drop the target from the training data
    if "TARGET" in df_train:
        train = df_train.drop(columns = ["TARGET"])
    else:
        train = df_train.copy()
        
    # Feature names
    features = list(train.columns)


    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))


    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(df_test)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    return train, test




target=data_train.copy()

def data_resampler(df_train, target):

    rsp = RandomUnderSampler()
    X_rsp, y_rsp = rsp.fit_resample(df_train, target["TARGET"])

    return X_rsp, y_rsp



def entrainement_LightBoost(X, y):

    # Configuration de la meilleure itération trouvée par le RandomizeSearchCV
    # Optimized n_estimator=1144
    clf_lgbm = LGBMClassifier(colsample_bytree=0.600170715692459, 
                              device='gpu',
                              learning_rate=0.02975841167356727, 
                              max_depth=7, 
                              n_estimators=600,
                              reg_lambda=2.156064880286573, 
                              subsample=0.629809210489369)

    clf_lgbm.fit(X, y)

    return clf_lgbm




def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn 


# On crée deux variables en attente qui deviendront
# des variables globales après l'initialisation de l'API.
# Ces variables sont utilisées dans plusieurs fonctions de l'API.
train = None
test = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test['SK_ID_CURR'][:50].values
id_client = pd.DataFrame(id_client)



import warnings
warnings.filterwarnings("ignore")

@app.route("/init_model", methods=["GET"])
def init_model():
    
    # On prépare les données
    df_train, df_test = features_engineering(data_train, data_test)
    df_train=df_train.drop(labels="Unnamed: 0",axis=1)
    df_test=df_test.drop(labels="Unnamed: 0",axis=1)

    print("Features engineering done")
    # On fait le préprocessing des données
    df_train, df_test = preprocesseur(df_train, df_test)

    # On transforme le dataset de test préparé en variabe
    # globale, car il est utilisé dans la fonction predict
    
    global train
    train = df_train.copy()

    global test 
    test = df_test.copy()

    print("Preprocessing done")
    # On fait un resampling des données d'entraînement
    X, y = data_resampler(df_train, data_train)
    print("Resampling done")

    # On entraîne le modèle et on le transforme en
    # variable globale pour la fonction predict
    
    global clf_lgbm
    clf_lgbm= entrainement_LightBoost(X, y)
    print("Training LightBoost done")

    global knn
    knn = entrainement_knn(df_train)
    print("Training knn done")
    
    print(test)
    print(train)

    return jsonify(["Initialisation terminée."])
    
# In[ ]:
#@app.route("/")
#def projet():
	#return "<p> Route projet <\p>"


# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    
    return id_client.to_json(orient='values')



# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target():

    df_target = data_train["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]
   
    
    dict_infos = {
       "status_famille" : data_client["NAME_FAMILY_STATUS"].squeeze(),
       "nb_enfant" : data_client["CNT_CHILDREN"].squeeze(),
       "age" : (data_client["DAYS_BIRTH"].values / -365),
       "revenus" : data_client["AMT_INCOME_TOTAL"].squeeze(),
       "montant_credit" : data_client["AMT_CREDIT"].squeeze(),
       "annuites" : data_client["AMT_ANNUITY"].squeeze(),
       "montant_bien" : data_client["AMT_GOODS_PRICE"].squeeze()
       }
    
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique
# situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])
     
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])
    print(test)
    
   
    data_client = test

    print(data_client)
    

    prediction = clf_lgbm.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)

@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test
    
    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return response



if __name__ == '__main__':
   app.run(port=5000)


