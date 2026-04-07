# Práctica 3: Detección de Tumores en Ecografías con Visión Artificial

Este proyecto utiliza modelos de Visión Artificial y Aprendizaje Profundo para analizar imágenes de ecografías médicas y clasificar la presencia de tumores como benignos o malignos y su posterior encapsulamiento en Bounding Boxes para mayor explicabilidad.

Destaca por su diseño modular orientado a objetos y el uso de **[uv](https://github.com/astral-sh/uv)** para una gestión del entorno y dependencias ultrarrápida y reproducible.

## Estructura del Proyecto

El código está organizado en distintos módulos para facilitar su lectura y mantenimiento:

```
├── 📁 utils
│   ├── 📁 bounding_boxes
│   │   ├── 🐍 F_Bounding_Boxes_Creator.py
│   │   └── 🐍 F_YOLO_Model.py
│   ├── 📁 classifiers
│   │   ├── 🐍 F_CNN_Classifier.py
│   │   ├── 🐍 F_Classifier_Manager.py
│   │   └── 🐍 F_ResNet_Classifier.py
│   ├── 📁 downloader
│   │   └── 🐍 F_Data_Downloader.py
│   ├── 📁 loaders
│   │   ├── 🐍 F_Fetal_Ultrasound_Dataloader.py
│   │   └── 🐍 F_Fetal_Ultrasound_Dataset.py
│   ├── 📁 metrics
│   │   └── 🐍 F_Metrics_Generator.py
│   ├── 📁 splitter
│   │   └── 🐍 F_Data_Splitter.py
│   └── 🐍 __init__.py
├── ⚙️ .gitignore
├── 📝 README.md
├── 🐍 main.py
├── ⚙️ pyproject.toml
└── 📄 uv.lock
```

## 1. Instalación de `uv`

Si aún no tienes el gestor de paquetes `uv` instalado en tu sistema, abre tu terminal y ejecuta el comando correspondiente a tu sistema operativo:

**Para macOS y Linux:**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

**Para Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

## 2. Configuración del Entorno

Como este proyecto utiliza `pyproject.toml` y `uv.lock`, la configuración es automática. Abre la terminal en la carpeta raíz del proyecto y ejecuta:

```bash
uv sync
```
*Este comando creará automáticamente el entorno virtual (`.venv`) e instalará las versiones exactas de las librerías (pandas, scikit-learn, xgboost, etc.) definidas en el archivo lock, garantizando que todo funcione a la primera.*

## 3. Ejecución del Código

Para ejecutar el programa principal, descargar el dataset automáticamente y entrenar los modelos, simplemente lanza:

```bash
uv run main.py
```

Una vez termine el proceso, podrás revisar la carpeta `results` (que se generará automáticamente en la raíz del proyecto) para ver los archivos de texto con las métricas cuantitativas y las imágenes `.png` con las matrices de confusión generadas por cada modelo de clasificación, además se generará una carpeta `YOLO` con imágenes `.png` pintando las predicciones del modelo.

> ⚠️ **IMPORTANTE: Credenciales de Kaggle**
>
> Este proyecto utiliza la API de Kaggle para descargar el dataset automáticamente. Para que funcione, necesitas tener configurado tu archivo de credenciales (`kaggle.json`).
>
> **Pasos para configurarlo:**
> 1. Inicia sesión en [Kaggle](https://www.kaggle.com/) y ve a los ajustes de tu cuenta (*Settings*).
> 2. Haz clic en **"Create New Token"** para descargar el archivo `kaggle.json`.
> 3. Guarda este archivo en la siguiente ruta dependiendo de tu sistema operativo:
>    - **Windows:** `C:\Users\<TuUsuario>\.kaggle\kaggle.json`
>    - **macOS / Linux:** `~/.kaggle/kaggle.json`
