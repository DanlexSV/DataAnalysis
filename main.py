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
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU.")


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


###### Generamos la primera tabla de datos comparando la cantidad de imagenes Humanas vs IA para saber si hay una diferencia de datos
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=train_df, hue="label", palette="coolwarm", legend=False)
plt.title("Label Distribution")
plt.xticks([0, 1], ["Human-Created", "AI-Generated"])
plt.show()


###### Comprobamos que las imagenes generadas por la IA esten sincronizadas con las generadas por Humanos ######
train_rng = random.Random()
train_random_state = train_rng.randint(0, len(train_df) - 1)
def show_images(df, label, num_images=5):
    sample_images = df[df["label"] == label].sample(num_images, random_state=train_random_state)["file_name"].values

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)  # Lee la imagen
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convierte BGR a RGB
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title("AI-Generated" if label == 1 else "Human-Created")

    plt.show()

show_images(train_df, label=1) # Mostramos las imagenes de IA
show_images(train_df, label=0) # Mostramos las imagenes de Humanos


###### Comprobamos que carguen las imagenes en los datos de prueba y tengan tanto generadas por humanos como por IA ######
test_rng = random.Random()
test_random_state = test_rng.randint(0, len(test_df) - 1)
def show_test_images(df, num_images=5):
    sample_images = df.sample(num_images, random_state=test_random_state)["id"].values

    plt.figure(figsize=(15, 5))
    for i, img_path in enumerate(sample_images):
        img = cv2.imread(img_path)  # Lee la imagen
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image {i+1}")  
    
    plt.show()

show_test_images(test_df, num_images=5)


###### Comparamos la distribucion de tamaño en las imagenes entre los datos de entrenamiento y de prueba ######
def get_image_dimensions(image_paths, sample_size=500):
    image_sizes = []

    for img_path in image_paths.sample(sample_size, random_state=test_random_state):
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            image_sizes.append((w, h))

    return pd.DataFrame(image_sizes, columns=["Width", "Height"])

train_size_df = get_image_dimensions(train_df["file_name"])
test_size_df = get_image_dimensions(test_df["id"])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x=train_size_df["Width"], y=train_size_df["Height"], alpha=0.5, ax=axes[0])
axes[0].set_title("Train Image Dimensions")
axes[0].set_xlabel("Width")
axes[0].set_ylabel("Height")

sns.scatterplot(x=test_size_df["Width"], y=test_size_df["Height"], alpha=0.5, ax=axes[1], color='red')
axes[1].set_title("Test Image Dimensions")
axes[1].set_xlabel("Width")
axes[1].set_ylabel("Height")

plt.tight_layout()
plt.show()
