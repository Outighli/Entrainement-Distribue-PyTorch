#  TD/TP : Entraînement Distribué avec PyTorch

##  Objectif du TP

Ce TP a pour objectif d'analyser l'impact de la distribution du calcul sur l'efficacité de l'entraînement d'un modèle de Deep Learning.

Nous comparons trois configurations :

-  CPU
-  1 GPU
-  2 GPU (DataParallel)

 Le modèle utilisé est **ResNet-50** appliqué au dataset **CIFAR-10**.

---

##  Pré-requis techniques

- Python 3.x
- PyTorch 2.0
- NumPy
- Matplotlib

###  Plateforme recommandée

- Kaggle Notebook avec **GPU T4 ×2**

###  Configuration GPU (Kaggle)

1. Aller dans **Settings**
2. Choisir **Accelerator → GPU**
3. Activer **GPU T4** (ou 2× T4 si disponible)
4. Vérifier **Phone Verification**

---

##  Dataset utilisé

- **CIFAR-10**
  - 10 classes d'images
  - Taille originale : **32×32**
  - Redimensionnée : **224×224**

>  **Pourquoi redimensionner ?** Parce que ResNet-50 nécessite des images de taille **224×224**.

---

##  Modèle utilisé

- **ResNet-50** pré-entraîné sur ImageNet

###  Adaptation du modèle

Remplacement de la couche finale afin d'adapter le modèle aux 10 classes de CIFAR-10 :

```python
model.fc = nn.Linear(2048, 10)
```

---

##  Structure du projet

```
Projet06/
│── projet06/               # Code source
│── presentation-p6.pptx
│── TP_projet06.pdf
│── README.md
```

---

##  Plan du TP

### 🔹 Partie 1 : Préparation et Exploration

**1. Préparation de l'environnement**
- Vérification des GPU disponibles
- Configuration du device

**2. Exploration du dataset**
- Chargement du dataset CIFAR-10
- Visualisation des images
- Analyse des classes

**3. Définition du modèle**
- Chargement de ResNet-50
- Modification de la couche finale

---

### 🔹 Partie 2 : Entraînement

#### Configuration A — CPU

- Entraînement sans GPU
- Sert de **baseline**
- Temps d'exécution très élevé (~2h 25min)

#### Configuration B — 1 GPU + AMP

- Utilisation de **CUDA**
- Activation de **Automatic Mixed Precision (AMP)**

```python
device = torch.device('cuda:0')
scaler = torch.cuda.amp.GradScaler()
```

>  **Avantages :**
> - Réduction du temps d'entraînement
> - Diminution de l'utilisation mémoire (float16)

####  Configuration C — DataParallel (2 GPU)

- Utilisation de **plusieurs GPU**
- Répartition des données entre les GPU
- Synchronisation automatique des gradients

```python
model = nn.DataParallel(model, device_ids=[0, 1])
model.to('cuda:0')
```

>  **Objectif :** Accélérer l'entraînement sans affecter la qualité du modèle.

---

##  Hyperparamètres

| Paramètre | Valeur |
|---|---|
| Batch size | 256 |
| Nombre d'epochs | 4 |
| Workers | 2 |
| Subset | 10 000 |

---

##  Résultats et Analyse

### 🔸 Comparaison des performances

| Configuration | Moy. / epoch | Temps total | Val Acc |
|---|---|---|---|
|  CPU | 2186.3 s | 145.8 min | — |
|  1 GPU (AMP) | 66.0 s | 4.4 min | 87.89 % |
|  DataParallel (2 GPU) | 54.4 s | 3.6 min | 88.45 % |

### 🔸 Visualisations produites

-  Courbes de Train Loss
-  Courbes de Validation Loss
-  Courbes d'Accuracy (Train & Validation)

---

##  Interprétation des résultats

-  L'utilisation du GPU réduit **considérablement** le temps d'entraînement (×33 vs CPU)
-  DataParallel permet une **accélération supplémentaire** (×1.2 vs 1 GPU)
-  Les performances (accuracy) restent **globalement similaires** entre 1 GPU et 2 GPU

>  Cela montre que la parallélisation **améliore la vitesse sans dégrader la qualité** de l'apprentissage.

---

##  Conclusion

Ce TP met en évidence que :

- Le **GPU** est essentiel pour les tâches de Deep Learning
- Le **parallélisme** permet de gagner un temps significatif
- **DataParallel** est une solution simple pour exploiter plusieurs GPU

---

##  Auteurs

-  OUARRAK Layla
-  RHAOUFAL Khadija
-  OUTIGHLI Sanae

> Étudiantes en **Intelligence Artificielle** — FSBM, Université Hassan II de Casablanca
