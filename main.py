import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import random
from sklearn.model_selection import train_test_split
from torchvision import transforms
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score ,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torchvision.models as models

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

###### Comprobamos si CUDA esta disponible o no ######
response = input("Do you want to CHECK if CUDA is available? (Y/N): ").strip().upper()

if response == "Y" and torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using GPU.")
else:
    device = torch.device("cpu")
    if response == "Y":
        print("CUDA is not available, using CPU.")
    else:
        print("CUDA check skipped, using CPU by default.")


###### Configuramos los Seeds para que sean reproducibles ######
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


###### Cargamos los Dataset e imprimimos la cantidad de datos obtenidos ######
base_dir = './dataset'
train_df = pd.read_csv("./dataset/train.csv")
test_df = pd.read_csv("./dataset/test.csv")

print("Train dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)
train_df.head()
test_df.head()

train_df = train_df.drop(columns=['Unnamed: 0'])


###### Actualizamos las columnas id de los Dataset de prueba, incluyendo la ruta absoluta en la ruta relativa existente
###### Actualizamos la columna file_name en los Dataset de entrenamiento para incluir la ruta absoluta a cada imagen.
test_df['id'] = test_df['id'].apply(lambda x: os.path.join(base_dir, x))
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(base_dir, x))


print(f"Train Data: {len(train_df)}")
print(f"Test Data: {len(test_df)}")


###### Comprobamos si no se han perdido datos o si se han duplicado algunos valores ######
print("Missing values in Train Dataset:\n", train_df.isnull().sum())
print("Missing values in Test Dataset:\n", test_df.isnull().sum())

print("Duplicate entries in Train Dataset:", train_df.duplicated().sum())


plt.figure(figsize=(6,4))
sns.countplot(x="label", data=train_df, palette="coolwarm")
plt.title("Label Distribution")
plt.xticks([0, 1], ["Human-Created", "AI-Generated"])
plt.show()
