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
print("\n")

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
train_df = pd.read_csv("./images_csv/train.csv")
test_df = pd.read_csv("./images_csv/test.csv")

print("Forma del dataset de entrenamiento:", train_df.shape, "\n")
print("Forma del dataset de prueba:", test_df.shape, "\n")
print(train_df.head(), "\n")
print(test_df.head(), "\n")

train_df = train_df.drop(columns=['Unnamed: 0'])


###### Actualizamos las columnas id de los Dataset de prueba
######incluyendo la ruta absoluta en la ruta relativa existente
test_df['id'] = test_df['id'].apply(lambda x: os.path.join(base_dir, x))
print(f"Train Data: {len(train_df)}")

###### Actualizamos la columna file_name en los Dataset de entrenamiento 
######para incluir la ruta absoluta a cada imagen.
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(base_dir, x))
print(f"Test Data: {len(test_df)} \n")


###### Comprobamos si no se han perdido datos o si se han duplicado algunos valores ######
print("Valores perdidos en el Train Dataset:\n", train_df.isnull().sum())
print("Valores perdidos en el Test Dataset:\n", test_df.isnull().sum())

print("Entradas duplicadas en el Train Dataset:", train_df.duplicated().sum(), "\n")


###### Generamos la primera grafica de comparación por la columna label
plt.figure(figsize=(6, 4))
sns.countplot(x="label", data=train_df, hue="label", palette="coolwarm", legend=False)
plt.title("Label Distribution")
plt.xticks([0, 1], ["Human-Created", "AI-Generated"])
plt.show()


###### Inspección visual de las muestras de imagenes en el train dataset ######
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


###### Inspección visual de imagenes en el test dataset ######
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
axes[0].grid(True)

sns.scatterplot(x=test_size_df["Width"], y=test_size_df["Height"], alpha=0.5, ax=axes[1], color='red')
axes[1].set_title("Test Image Dimensions")
axes[1].set_xlabel("Width")
axes[1].set_ylabel("Height")
axes[1].grid(True)

plt.tight_layout()
plt.show()


###### Funcion para obtener las dimensiones entre IA vs Humanos
def get_image_dimensions(image_paths, sample_size=500):
    image_sizes = []

    for img_path in image_paths.sample(sample_size, random_state=train_random_state):
        img = cv2.imread(img_path)
        if img is not None:
            h, w, _ = img.shape
            image_sizes.append((w, h))

    return pd.DataFrame(image_sizes, columns=["Width", "Height"])

ai_images = train_df[train_df["label"] == 1]["file_name"]
human_images = train_df[train_df["label"] == 0]["file_name"]

ai_size_df = get_image_dimensions(ai_images)
human_size_df = get_image_dimensions(human_images)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AI images
sns.scatterplot(x=ai_size_df["Width"], y=ai_size_df["Height"], alpha=0.5, ax=axes[0], color='red')
axes[0].set_title("AI-Generated Image Dimensions")
axes[0].set_xlabel("Width")
axes[0].set_ylabel("Height")
axes[0].grid(True)

# Human images
sns.scatterplot(x=human_size_df["Width"], y=human_size_df["Height"], alpha=0.5, ax=axes[1], color='blue')
axes[1].set_title("Human-Created Image Dimensions")
axes[1].set_xlabel("Width")
axes[1].set_ylabel("Height")
axes[1].grid(True)

plt.tight_layout()
plt.show()


###### Funcion para comparar la intensidad de pixeles entre IA vs Humanos
def plot_pixel_intensity_side_by_side(img_path1, img_path2, title1="AI-Generated", title2="Human-Created"):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(img1.ravel(), bins=256, color="red", alpha=0.7)
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{title1} - Pixel Intensity")
    axes[0].grid(True)

    axes[1].hist(img2.ravel(), bins=256, color="blue", alpha=0.7)
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title2} - Pixel Intensity")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

sample_ai = train_df[train_df["label"] == 1]["file_name"].sample(1).values[0]
sample_human = train_df[train_df["label"] == 0]["file_name"].sample(1).values[0]

plot_pixel_intensity_side_by_side(sample_ai, sample_human)


###### Calculo de la media y varianza de la intensidad de pixeles
def compute_intensity_stats(file_paths):
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for fp in file_paths:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        pixels = img.ravel()
        total_sum    += pixels.sum()
        total_sumsq  += (pixels ** 2).sum()
        total_count  += pixels.size

    mean = total_sum / total_count
    var  = (total_sumsq / total_count) - (mean ** 2)
    return mean, var

###### Extrae listas de rutas
ai_paths    = train_df.loc[train_df.label == 1, "file_name"].tolist()
hum_paths   = train_df.loc[train_df.label == 0, "file_name"].tolist()

###### Calcula estadísticas
mean_ai, var_ai   = compute_intensity_stats(ai_paths)
mean_hum, var_hum = compute_intensity_stats(hum_paths)

print(f"IA-generated images:   media intensidad = {mean_ai:.2f}, varianza = {var_ai:.2f}")
print(f"Human-created images:  media intensidad = {mean_hum:.2f}, varianza = {var_hum:.2f}")


###### Funcion para observar la distribucion de colores que tienen las imagenes
###### Y comparar la distribucion entre IA vs Humanos
def plot_color_distribution_side_by_side(img_path1, img_path2, title1="AI-Generated", title2="Human-Created"):
    img1 = cv2.imread(img_path1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    img2 = cv2.imread(img_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    r1, g1, b1 = cv2.split(img1)
    r2, g2, b2 = cv2.split(img2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AI-Generated Image Histogram
    axes[0].hist(r1.ravel(), bins=256, color="red", alpha=0.5, label="Red")
    axes[0].hist(g1.ravel(), bins=256, color="green", alpha=0.5, label="Green")
    axes[0].hist(b1.ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"{title1} - Color Distribution")
    axes[0].legend()
    axes[0].grid(True)

    # Human-Created Image Histogram
    axes[1].hist(r2.ravel(), bins=256, color="red", alpha=0.5, label="Red")
    axes[1].hist(g2.ravel(), bins=256, color="green", alpha=0.5, label="Green")
    axes[1].hist(b2.ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
    axes[1].set_xlabel("Pixel Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title2} - Color Distribution")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

sample_ai = train_df[train_df["label"] == 1]["file_name"].sample(1).values[0]
sample_human = train_df[train_df["label"] == 0]["file_name"].sample(1).values[0]

plot_color_distribution_side_by_side(sample_ai, sample_human)

