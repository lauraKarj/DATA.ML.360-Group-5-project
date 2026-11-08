# DATA.ML.360-Group-5-project
Recommender Systems project work for the course of Tampere University course DATA.ML.360

Team of creators, Group 5: Laura Karjalainen, Tommi Lamberg and Kush Aswani

Part 1 report and results can be found in the Repository: report/part-1/

# Requirements

- Requirement is Python 3.13
- Project is coded by using Python and requirements for the project is included in requirements.txt file.

# Clone the project

1. Open terminal
2. Select and go to forder where you want to save the project
3. Clone Git repository by
    ```
   Git clone https://github.com/lauraKarj/DATA.ML.360-Group-5-project.git
   ```
4. Go to folder 
    ```
   cd DATA.ML.360-Group-5-project
   ```

# Data and preparation

Dataset used for the project is MovieLens 1M by GroupLens.
Data is available in https://grouplens.org/datasets/movielens/1m/

For the project, data isn't included in the project folder and it should be retrieved before running the project.
That is due to GroupLens and dataset's licence (from ml-1m/README). 
Instructions for data set up:

1. Click https://files.grouplens.org/datasets/movielens/ml-1m.zip which will download the data.
2. Unzip the loaded forder.
3. Open the folder, copy three datasets inside it and paste those in project folder ml-1m/
4. Now the project forder ml-1m/ structure should look like following:
    ```
    DATA.ML.360-GROUP-5-project
    ├── figures/
    ├── ml-1m/
    │   └── movies.dat
    │   └── ratings.dat
    │   └── README
    │   └── users.dat
    ```

# Install dependencies

## 1. Virtual environment can be installed by

Windows PowerShell:

```py -3.13 -m venv venv``` 

Linux / macOS:

```python3.13 -m venv venv```

## 2. Activation

Windows PowerShell

```venv\Scripts\activate.bat```

Linux / macOS

```source venv/bin/activate``` 

## 3. Install required dependencies:

Windows: 

```venv\Scripts\python.exe -m pip install -r requirements.txt```

Linux / macOS:

```./venv/bin/python -m pip install -r requirements.txt```


# How to run the project

In the terminal, you should be on folder DATA.ML.360-Group-5-project/
```
cd DATA.ML.360-Group-5-project
```

Make sure the MovieLens 1M dataset is downloaded and located in the ml-1m directory inside the project root.
If not, see Data and preparation part above.

In folder run following:

Windows

```venv\Scripts\python.exe run_all.py --data-dir ml-1m```

Linux / macOS:

```./venv/bin/python run_all.py --data-dir ml-1m```

The script automatically (Part 1):
- Performs data preparation and generates distribution plots.
- Runs the Item-KNN model and evaluation.
- Runs the Most Popular model and evaluation.
- Saves CSV results in the results directory.
- Saves generated figures in the figures directory.

# Structure
```
DATA.ML.360-GROUP-5-project
├── figures/ # Plots saved as png files
├── ml-1m/   # Data used
├── results/ # Results saved as CSV files
│
├── .gitignore 
├── data_preparation.py
├── evaluation.py
├── item_knn.py
├── most_popular.py
├── README.md 
├── requirements.txt # Includes requirements
└── run_all.py # File that runs the whole project
```
