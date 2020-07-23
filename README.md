# stores

Mini Competition - Rossman Challenge

This is a repo to predict Sales data from the Rossman Challenge


[Optional: install a new environment, activate it and install the requirements.txt]

```
conda create --name rossmann  python=3.7
conda activate rossmann
pip install -r requirements.txt
```

To run:

1. Clone the repo https://github.com/elenche95/stores in your local pc and check the requirements.txt

2. Execute from the main folder (./stores) in the terminal: 

```
python data.py --test 1
```

Make sure that "./data/test.csv" was created

3. Download from the following link the Random Forest regressor model that was trained with the train.csv data

https://mega.nz/file/PSo0kSjJ#jFiRslRxwz18bcWeogufvqgr4gUcRLirD-K-yra0Hsk 

4. Unzip the file and store dtree_final.joblib in the main folder

5. From terminal run:  

```
python express.py
```

Once the script gets fully executed, it will print the RMSPE in the terminal for the test set.
