import scipy.io
import pandas as pd

# 1. Carica il file .mat (sostituisci col nome esatto del tuo file)
mat_data = scipy.io.loadmat('Dati/colon.mat')

# 2. Ispeziona le chiavi contenute nel file
print("Chiavi presenti nel file .mat:", mat_data.keys())

# Nelle collezioni HDLSS/scikit-feature, la matrice dei dati è 'X' e le etichette 'Y' (o 'y')
X = mat_data['X']  # Matrice delle feature (es. 62 righe x 2000 colonne)
Y = mat_data['Y']  # Vettore della classe (es. 62 righe x 1 colonna)

# Se Y è bidimensionale (62, 1), rendilo piatto 1D
if Y.ndim > 1:
    Y = Y.ravel()

# 3. Crea il DataFrame Pandas
# Nomina le colonne delle feature come gene_1, gene_2, ... e aggiungi la colonna finale 'target'
feature_names = [f"gene_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = Y

# 4. Salva il file definitivo in CSV pronto per SynthBridge
df.to_csv('dati/colon.csv', index=False)
print("Conversione completata! Creato dati/colon.csv con dimensioni:", df.shape)
