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


def compute_intensity_stats(file_paths):
    """
    Calculo de la media y desviación estándar de la intensidad de píxeles
    """
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0

    for fp in file_paths:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        pixels = img.ravel()
        total_sum   += pixels.sum()
        total_sumsq += (pixels ** 2).sum()
        total_count += pixels.size

    mean = total_sum / total_count
    var  = (total_sumsq / total_count) - (mean ** 2)
    stddev = np.sqrt(var)
    return mean, stddev

###### Extrae listas de rutas
ai_paths  = train_df.loc[train_df.label == 1, "file_name"].tolist()
hum_paths = train_df.loc[train_df.label == 0, "file_name"].tolist()

###### Calcula estadísticas
mean_ai, std_ai   = compute_intensity_stats(ai_paths)
mean_hum, std_hum = compute_intensity_stats(hum_paths)

print(f"IA-generated images:   intensidad media = {mean_ai:.2f}, desviación estándar = {std_ai:.2f}")
print(f"Human-created images:  intensidad media = {mean_hum:.2f}, desviación estándar = {std_hum:.2f}")


def plot_color_channels_expanded(img_path, title_prefix="Image"):
    """
    Dibuja 4 subgráficos para la imagen en img_path:
     - Solo canal Rojo
     - Solo canal Verde
     - Solo canal Azul
     - Canales combinados (RGB)
    El título de cada subgráfico se construye con title_prefix.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img)

    # Creamos una figura con 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # 1) Solo Rojo
    axes[0, 0].hist(r.ravel(), bins=256, color="red", alpha=0.7)
    axes[0, 0].set_title(f"{title_prefix} – Red Channel")
    axes[0, 0].set_xlabel("Pixel Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True)

    # 2) Solo Verde
    axes[0, 1].hist(g.ravel(), bins=256, color="green", alpha=0.7)
    axes[0, 1].set_title(f"{title_prefix} – Green Channel")
    axes[0, 1].set_xlabel("Pixel Value")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True)

    # 3) Solo Azul
    axes[1, 0].hist(b.ravel(), bins=256, color="blue", alpha=0.7)
    axes[1, 0].set_title(f"{title_prefix} – Blue Channel")
    axes[1, 0].set_xlabel("Pixel Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(True)

    # 4) Canales combinados (RGB)
    axes[1, 1].hist(r.ravel(), bins=256, color="red", alpha=0.5, label="Red")
    axes[1, 1].hist(g.ravel(), bins=256, color="green", alpha=0.5, label="Green")
    axes[1, 1].hist(b.ravel(), bins=256, color="blue", alpha=0.5, label="Blue")
    axes[1, 1].set_title(f"{title_prefix} – Combined RGB")
    axes[1, 1].set_xlabel("Pixel Value")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

sample_ai = train_df[train_df["label"] == 1]["file_name"].sample(1).values[0]
sample_human = train_df[train_df["label"] == 0]["file_name"].sample(1).values[0]

plot_color_channels_expanded(sample_ai, title_prefix="IA-generated")
plot_color_channels_expanded(sample_human, title_prefix="Human-created")


def compare_random_color_pairs(df, rng=None):
    """
    Toma el Train Dataset para obtener dos imágenes aleatorias de un mismo índice relativo
    en ambos grupos (IA y Humanos), y para cada imagen dibuja cuatro histogramas:
      - Canal Rojo
      - Canal Verde
      - Canal Azul
      - Combinado RGB
    """

    if rng is None:
        rng = random.Random()

    ia_paths = df.loc[df.label == 1, "file_name"].tolist()
    hu_paths = df.loc[df.label == 0, "file_name"].tolist()

    min_len = min(len(ia_paths), len(hu_paths))
    if min_len < 2:
        raise ValueError("Se necesitan al menos 2 imágenes en cada grupo (IA y Humanos)")

    # Elegir dos índices aleatorios (mismos para IA y Humanos)
    idxs = rng.sample(range(min_len), 2)

    # Extraer las dos imágenes de IA y las dos de Humanos
    ia_img1, ia_img2   = [ia_paths[i] for i in idxs]
    hu_img1, hu_img2   = [hu_paths[i] for i in idxs]

    # Para cada imagen IA, dibujar los 4 histogramas
    plot_color_channels_expanded(ia_img1, title_prefix="IA-generated #1")
    plot_color_channels_expanded(ia_img2, title_prefix="IA-generated #2")

    # Para cada imagen Humana, dibujar los 4 histogramas
    plot_color_channels_expanded(hu_img1, title_prefix="Human-created #1")
    plot_color_channels_expanded(hu_img2, title_prefix="Human-created #2")

compare_random_color_pairs(train_df, rng=train_rng)


def compute_color_stats_all(file_paths):
    """
    Recorre todas las imágenes de file_paths y acumula:
      - total_sum[c]: suma de todos los valores del canal c
      - total_sumsq[c]: suma de cuadrados de todos los valores del canal c
      - total_count: número total de píxeles procesados (mismo para cada canal)

    Luego calcula media y desviación estándar por canal.
    Devuelve un DataFrame con columnas ['Channel','Mean','StdDev'].
    """
    total_sum   = np.zeros(3, dtype=np.float64)
    total_sumsq = np.zeros(3, dtype=np.float64)
    total_count = 0

    for fp in file_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        pixels = img_rgb.reshape(-1, 3)

        total_sum   += pixels.sum(axis=0)
        total_sumsq += (pixels ** 2).sum(axis=0)
        total_count += pixels.shape[0]

    if total_count == 0:
        raise ValueError("No hay píxeles válidos en las rutas proporcionadas.")

    means = total_sum / total_count
    variances = total_sumsq / total_count - means**2
    stddevs = np.sqrt(variances)

    return pd.DataFrame({
        'Channel': ['R', 'G', 'B'],
        'Mean':    means,
        'StdDev':  stddevs
    })

ai_paths = train_df.loc[train_df.label == 1, "file_name"].tolist()
hu_paths = train_df.loc[train_df.label == 0, "file_name"].tolist()

ai_stats_df = compute_color_stats_all(ai_paths)
hu_stats_df = compute_color_stats_all(hu_paths)

print("\n=== IA-generated Color Stats ===\n", ai_stats_df)
print("\n=== Human-created Color Stats ===\n", hu_stats_df)


def per_image_channel_means(file_paths):
    """
    Para cada ruta en file_paths:
      - Carga la imagen en RGB
      - Calcula la media de R, G y B
    Devuelve un DataFrame de forma (n_imágenes × 3 canales).
    """
    data = {'R': [], 'G': [], 'B': []}
    for fp in file_paths:
        img = cv2.imread(fp)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Media de cada canal
        data['R'].append(rgb[:, :, 0].mean())
        data['G'].append(rgb[:, :, 1].mean())
        data['B'].append(rgb[:, :, 2].mean())
    return pd.DataFrame(data)

ai_means_df = per_image_channel_means(ai_paths)
hu_means_df = per_image_channel_means(hu_paths)


def plot_channel_means_boxplots(ai_df, hu_df):
    """
    Dibuja dos boxplots lado a lado:
      - ax[0]: distribución de medias por canal para IA
      - ax[1]: distribución de medias por canal para Humanos
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # IA
    ax[0].boxplot(
        [ai_df['R'], ai_df['G'], ai_df['B']],
        tick_labels=['R','G','B']
    )
    ax[0].set_title('IA-generated: Channel Means')
    ax[0].set_ylabel('Mean Pixel Value')
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Humanos
    ax[1].boxplot(
        [hu_df['R'], hu_df['G'], hu_df['B']],
        tick_labels=['R','G','B']
    )
    ax[1].set_title('Human-created: Channel Means')
    ax[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

plot_channel_means_boxplots(ai_means_df, hu_means_df)


def summarize_means_df(df):
    """
    Dado un DataFrame de medias por imagen y canal,
    devuelve un resumen con:
      - Mean of Means
      - StdDev of Means
    (La varianza se omite intencionalmente.)
    """
    return pd.DataFrame({
        'Channel':         ['R','G','B'],
        'Mean of Means':   df.mean().values,
        'StdDev of Means': df.std().values
    })


ai_summary = summarize_means_df(ai_means_df)
hu_summary = summarize_means_df(hu_means_df)

print("\n=== IA-generated Summary ===\n", ai_summary)
print("\n=== Human-created Summary ===\n", hu_summary)
