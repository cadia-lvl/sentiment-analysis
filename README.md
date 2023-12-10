# Viðhorfsgreining á íslenskum texta

(Sentiment-Analysis on Icelandic text)

# Instructions

## Machine-translate

This section provides instructions for using the machine translation scripts included in this project: `translate_google.py` and `translate_mideind.py`. These scripts are used for translating text data into Icelandic for sentiment analysis.

#### Using `translate_google.py`

##### Overview

`translate_google.py` is a Python script for translating text data using Google's translation service. It translates reviews from the `"IMDB-Dataset.csv"` file located in the `Datasets` directory and saves the translated text in a new file. The script uses multithreading to enhance performance and includes error handling for translation failures.

##### Prerequisites

-   Python 3.x
-   Pandas library
-   `googletrans` version 3.1.0a0
-   Other dependencies: `concurrent.futures`, `threading`, `logging`

##### Installation

1. Ensure Python 3.x is installed.
2. Install the required Python packages:
    - pip install pandas
    - pip install googletrans==3.1.0a0

##### Usage

1. Run the script:

    - python translate_google.py

2. Select the CSV file containing the text to be translated when prompted. The file should have columns named 'review' and 'sentiment'.
3. The script will process the data and output two files in the `Datasets` directory:
    - `IMDB-Dataset-GoogleTranslate.csv`: Contains translated reviews and sentiments.
    - `failed-IMDB-Dataset-GoogleTranslate.csv`: Logs failed translation attempts.

##### Custom Dataset

To use a different dataset:

-   Place your CSV dataset in the `Datasets` directory.
-   The dataset should have 'review' and 'sentiment' columns.
-   Modify the script if your dataset columns have different names.
-   Modify the script's `dataset` variable to match your dataset's filename.

#### Using `translate_mideind.py`

##### Overview

`translate_mideind.py` is a Python script for translating text data using the `"mideind/nmt-doc-en-is-2022-10"` model. It translates reviews from the `"IMDB-Dataset.csv"` file in the `Datasets` directory and saves the translated text in a new file.

##### Prerequisites

-   Python 3.x
-   `transformers` and `torch` libraries
-   Pandas library
-   Other dependencies: `re`, `logging`

##### Installation

1. Ensure Python 3.x is installed.
2. Install the required Python packages:
    - pip install transformers torch pandas

##### Usage

1. Run the script:
    - python translate_mideind.py
2. Select the folder containing the translation model when prompted.
3. Select the CSV file containing the text to be translated. The file should have columns named 'review' and 'sentiment'.
4. The script will process the data and output two files in the `Datasets` directory:
    - `IMDB-Dataset-MideindTranslate.csv`: Contains translated reviews and sentiments.
    - `failed-IMDB-Dataset-MideindTranslate.csv`: Logs failed translation attempts.

##### Custom Dataset

To use a different dataset:

-   Place your CSV dataset in the `Datasets` directory.
-   The dataset should have 'review' and 'sentiment' columns.
-   Modify the script if your dataset columns have different names.
-   Modify the script's `dataset` variable to match your dataset's filename.

## Process

### Processing Icelandic Text

This section provides instructions for using the `process.py` script, which performs text normalization and preprocessing for Icelandic text using IceNLP.

#### Prerequisites

-   Python 3.x
-   Pandas library
-   IceNLP tool (https://github.com/hrafnl/icenlp)
-   Other dependencies: `multiprocessing`, `os`, `string`, `sys`, `time`, `tkinter`, `re`, `joblib`, `nefnir`

#### Installation

1. Ensure Python 3.x is installed.
2. Install the required Python packages:
    - pip install pandas joblib nefnir
3. Download IceNLP from [IceNLP GitHub Repository](https://github.com/hrafnl/icenlp) and extract it.

#### Usage

1. Run the script:
    - python process.py
2. When prompted, select the `icetagger.bat` file located in the extracted IceNLP directory (IceNLP-1.5.0\IceNLP\bat\icetagger).
3. Ensure the dataset file (`IMDB-Dataset-MideindTranslate.csv`) is located in the `Datasets` directory relative to the script.
4. The script will process the dataset and output the processed data to `Datasets/IMDB-Dataset-MideindTranslate-processed-nefnir.csv`.

#### Custom Dataset

To use a different dataset:

-   Place your CSV dataset in the `Datasets` directory.
-   The dataset should have 'review' and 'sentiment' columns.
-   Modify the `dataset_path` variable in the script to match your dataset's filename.

### Processing English Text

This section provides instructions for using the `process_eng.py` script, which performs text normalization and preprocessing for English text.

#### Prerequisites

-   Python 3.x
-   Pandas library
-   NLTK library
-   Other dependencies: `os`, `time`, `re`, `joblib`

#### Installation

1. Ensure Python 3.x is installed.
2. Install the required Python packages:
    - pip install pandas nltk joblib
3. Download necessary NLTK data:
    - python -m nltk.downloader punkt stopwords wordnet

#### Usage

1. Ensure the dataset file (`IMDB-Dataset.csv`) is located in the `Datasets` directory relative to the script.
2. Run the script:
    - python process_eng.py
3. The script will process the dataset and output the processed data to `Datasets/IMDB-Dataset-Processed.csv`.

#### Custom Dataset

To use a different dataset:

-   Place your dataset in the `Datasets` directory.
-   The dataset should be in CSV format with a 'review' column.
-   Modify the `dataset_path` variable in the script to match your dataset's filename.

## Baseline Classifiers

## Transformer Models

This section provides instructions for using the `train.py` script, which trains a transformer model for sentiment analysis.

### Prerequisites

-   Python 3.x
-   Transformers library
-   PyTorch
-   Pandas library
-   Scikit-learn library
-   Other dependencies: `os`, `time`, `numpy`

### Installation

1. Ensure Python 3.x is installed.
2. Install the required Python packages:
    - pip install transformers torch pandas scikit-learn

### Usage

1. Place the dataset file (default: `"IMDB-Dataset-GoogleTranslate.csv"`) in the `Datasets` directory relative to the script.
2. Modify the script if you want to use a different pre-trained model or dataset.
3. Run the script:
    - python train.py
4. The script will train the model using the specified dataset and save the trained model and tokenizer in the `Models` directory.

### Custom Dataset

To use a different dataset:

-   Place your dataset in the `Datasets` directory.
-   The dataset should be in CSV format with 'review' and 'sentiment' columns.
-   Modify the `dataset_path` variable in the script to match your dataset's filename.

# Style

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# License

MIT

# Authors

Ólafur Aron Jóhannsson \
Eysteinn Örn \
Birkir Arndal
