from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.filterbank import filterbank
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.fs.mibif import MIBIF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import pandas as pd
from bciflow.modules.analysis.metric_functions import accuracy
import numpy as np
import matplotlib.pyplot as plt  #Import necessário para o gráfico

# Carrega o dataset
dataset = cbcic(subject=1, path=r'C:\Users\Kenzo\Desktop\BCI\Sujeito 1')

# Inspeção inicial
print("🔹 Chaves do dataset:", dataset.keys())
print("🔹 Formato de X:", dataset['X'].shape)  # (trials, bands, electrodes, time)
print("🔹 Formato de y:", dataset['y'].shape)
print("🔹 Valores únicos em y:", np.unique(dataset['y']))

# Exemplo de sinal original
print("\n🔸 EEG original [trial 0, banda 0, eletrodo 0] (10 amostras):")
print(dataset['X'][0, 0, 0, :10])
before = dataset['X'][0, 0, 0, :10]
class StandardScalerEEG:
    def __init__(self):
        pass

    def fit(self, eegdata: dict):
        X = eegdata['X']
        self.median = np.median(X)
        self.log_Median = np.log(self.median)
        
       
        return self

    def transform(self, eegdata: dict):
        X = eegdata['X']
        X_results = X - 4*(self.median - self.log_Median)
        eegdata['X'] = X_results
        return eegdata

    def fit_transform(self, eegdata: dict):
        return self.fit(eegdata).transform(eegdata)

# Aplica a normalização
scaler = StandardScalerEEG()
dataset_scaled = scaler.fit_transform(dataset)

# Exemplo de sinal normalizado
print("\n🔸 EEG normalizado [trial 0, banda 0, eletrodo 0] (10 amostras):")
print(dataset_scaled['X'][0, 0, 0, :10])

# Comparação visual (plot)
after = dataset_scaled['X'][0, 0, 0, :10]

trial_idx = 0
band_idx = 0
electrode_idx = 0
num_samples_to_plot = 500

before_plot = dataset['X'][trial_idx, band_idx, electrode_idx, :num_samples_to_plot]
after_plot = dataset_scaled['X'][trial_idx, band_idx, electrode_idx, :num_samples_to_plot]

plt.figure(figsize=(14, 6))

plt.plot(before_plot, label='Sinal Original', color='#1f77b4', linestyle='-', linewidth=1.5)
plt.plot(after_plot, label='Sinal Normalizado', color='#ff7f0e', linestyle='--', linewidth=1.5)

plt.legend(loc='upper right', fontsize=10)
plt.title(f'Comparação do Sinal EEG (Trial {trial_idx}, Banda {band_idx}, Eletrodo {electrode_idx}) Antes e Depois da Normalização', fontsize=14)
plt.xlabel('Tempo (amostras)', fontsize=12)
plt.ylabel('Amplitude', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)

plt.tight_layout()
plt.show()
