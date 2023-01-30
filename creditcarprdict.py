import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import time

print("\nReading Data from csv...\n\n")
time.sleep(2)
df = pd.read_csv('./UCI_Credit_Card.csv')
# print(df.shape) 
# print(df.head())

#Data cleaning 
#separet X and Y 
print("Data Processing...\n\n")
time.sleep(2)
X =df.drop(columns='default.payment.next.month',axis=1)
Y = df['default.payment.next.month']
# print(Y.shape)
# print(Y)
# print(X.shape)

#Data Preprocessing 
# scaler = StandardScaler()
# scaler.fit(X)
# standerard_data = scaler.transform(X)


#train test the data 
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,stratify=Y,random_state=1)
# print(X_train.shape)
# print(X_test.shape)

#Model Training 
print("Train the model...\n\n")
time.sleep(2)
model = svm.SVC(gamma='auto')
model.fit(X_train,Y_train)
# predict_data = model.predict(X_test)
# print(predict_data)

Score = model.score(X_test,Y_test)
# print(Score)


#decision Tree Prediction 
decision = DecisionTreeClassifier()

decision.fit(X_train,Y_train)
scr = decision.score(X_test,Y_test)
# print(scr)



#input System Check 

print("Done...\n\n")
time.sleep(2)


print("*"*10+"Creadit Card Prdiction"+"*"*10)
id = int(input("Id: "))
limit_bal = int(input("Limit_bal: "))
sex = int(input("Sex(Mail,Female(1,2)):"))
education = int(input("Eduction: "))
marrige = int(input("Marrige Status: "))
age = int(input("Age: "))
pay0 = int(input("Pay 0: "))
pay2 = int(input("Pay 2: "))
pay3 = int(input("Pay 3: "))
pay4 = int(input("Pay 4: "))
pay5 = int(input("Pay 5: "))
pay6 = int(input("Pay 6: "))
bhil_amt1 = int(input("Bhill_amt1: "))
bhil_amt2 = int(input("Bhill AMT2: "))
bhil_amt3 = int(input("Bhill AMT3: "))
bhil_amt4 = int(input("Bhill AMT4: "))
bhil_amt5 = int(input("Bhill AMT5: "))
bhil_amt6 = int(input("Bhill AMT6: "))
pay_atm1 = int(input("Pay_Atm1: "))
pay_atm2 = int(input("Pay_Atm2: "))
pay_atm3 = int(input("Pay_Atm3: "))
pay_atm4 = int(input("Pay_Atm4: "))
pay_atm5 = int(input("Pay_Atm5: "))
pay_atm6 = int(input("Pay_Atm6: "))

input_file = (id,limit_bal,sex,education,marrige,age,pay0,pay2,pay3,pay4,pay5,pay6,bhil_amt1,bhil_amt2,bhil_amt3,bhil_amt4,bhil_amt5,bhil_amt6,pay_atm1,pay_atm2,pay_atm3,pay_atm4,pay_atm5,pay_atm6)

convet_np = np.asarray(input_file)
reshaped_np = convet_np.reshape(1,-1)
svm_ans = model.predict(reshaped_np)
decision_ans = decision.predict(reshaped_np)

print("Predict Your result..\n\n")
time.sleep(2)
if(svm_ans[0]==1):
    print("Yes, Default payment Next Month")
else:
    print("NO, Default Payment to the next Month")
    

