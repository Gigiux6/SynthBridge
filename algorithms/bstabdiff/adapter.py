import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from bstabdiff import BSTabDiff

def generate(data, n_rows, params, info):
    """
    Adapter definitivo per BSTabDiff con gestione flessibile del return di sample().
    """
    seed = info.get("seed", 42)
    np.random.seed(seed)
    
    if "progress" in info and callable(info["progress"]):
        info["progress"](0.1, "Preprocessing dei dati e gestione colonne categoriche...")

    df_train = data.copy()
    label_encoders = {}
    categorical_cols = info.get("categorical_columns", [])
    
    if not categorical_cols:
        categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col].astype(str))
        label_encoders[col] = le

    if "progress" in info and callable(info["progress"]):
        info["progress"](0.3, "Inizializzazione e fit di BSTabDiff...")

    model = BSTabDiff()
    model.fit(df_train)

    if "progress" in info and callable(info["progress"]):
        info["progress"](0.8, "Generazione campioni sintetici...")

    # Campionamento dal modello
    raw_output = model.sample(n_samples=n_rows)
    
    # Se il modello restituisce una tupla (es. (synthetic_df, mask)), prende il primo elemento
    if isinstance(raw_output, tuple):
        synthetic_df = raw_output[0]
    else:
        synthetic_df = raw_output
        
    # Se l'output è un array numpy, riconverti in DataFrame con le colonne originali
    if isinstance(synthetic_df, np.ndarray):
        synthetic_df = pd.DataFrame(synthetic_df, columns=data.columns)

    # Decodifica delle variabili categoriche
    for col, le in label_encoders.items():
        if col in synthetic_df.columns:
            indices = np.clip(np.round(synthetic_df[col].values).astype(int), 0, len(le.classes_) - 1)
            synthetic_df[col] = le.inverse_transform(indices)

    if "progress" in info and callable(info["progress"]):
        info["progress"](1.0, "Generazione completata con successo.")

    return synthetic_df
