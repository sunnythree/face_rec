### face_arec
***this is a project for face recognization***  
I split lfw dataset into two parts, first part for training and second part for test.
and I get 98.14% accuracy on the test part.

#### train  
```
python3.6 Train.py -b 100 -l 0.001 -e 2000 -s 500  
```
if you want train from last time, just add -p True
```
python3.6 Train.py -b 100 -l 0.001 -e 2000 -s 500  -P true
```
#### test on  eval dataset
```
python3.6 TestHard.py
