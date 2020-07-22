# stores

Mini Competition - Rossman Challenge

This is a repo to predict Sales data from the Rossman Challenge

To run:

0. Clone the repo https://github.com/elenche95/stores and check the requirements.txt

1. Execute from the main folder in the terminal "python data.py --test 1" (make sure that "./data/test.csv" exists)

2. Download from the following link the Random Forest regressor model that was trained with the train.csv data

https://mega.nz/file/PSo0kSjJ#jFiRslRxwz18bcWeogufvqgr4gUcRLirD-K-yra0Hsk 

3. Unzip the file and store dtree_final.joblib in the main folder

4. From terminal run: "python express.py", once the script gets fully executed, it will print the RMSPE in the terminal for the test set.
