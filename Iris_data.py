'''
                                                                                                Student Name: Prachi Goel
                                                                                                Student ID: 1001234789
                                                                            Project 1 : Classification of Iris data using Linear regression'''

from numpy import *
from random import shuffle
import plotly.plotly as py
import plotly.offline as po
import plotly.graph_objs as go


#variables used in the project
X=[]
Y=[]
Sample_attributes_X=[]
Sample_labels_Y=[]
class_1 = {"True_Positive": 0, "True_Negative": 0, "False_Positive": 0, "False_Negative": 0}
class_2 = {"True_Positive": 0, "True_Negative": 0, "False_Positive": 0, "False_Negative": 0}
class_3 = {"True_Positive": 0, "True_Negative": 0, "False_Positive": 0, "False_Negative": 0}
Accuracy_class_1 = []
Accuracy_class_2 = []
Accuracy_class_3 = []

#reading the data set from the CSV file and storing in list X and Y where X contains the atrributes and Y contains the classes
def reading_csv():
    Sample_data=raw_input("Plese enter the location of data set to be analysed:" )
    with open(Sample_data,'r') as data:
        for line in data:
            line=line.strip()
            line=line.split(",")
            adding_one=[1]
            X.append(adding_one+line[0:4])
            Y.append(line[4])
    for items in X:
        Sample_attributes_X.append([float(item) for item in items])
    for items in Y:
        if items=="Iris-setosa":
            Sample_labels_Y.append([1])
        elif items=="Iris-versicolor":
            Sample_labels_Y.append([2])
        elif items=="Iris-virginica":
            Sample_labels_Y.append([3])
    return Sample_attributes_X,Sample_labels_Y

#Training the data using Linear regression and k fold cross validation where k = 15
def training_classifier(Sample_attributes_X,Sample_labels_Y):
    indice=[i for i in range(150)]
    j = 0

    #shuffling of data such that random data is selected from the data set
    shuffle_Y_N = raw_input("Please enter Y to shuffle the data else press any other key: ")
    if shuffle_Y_N == "Y":
        shuffle(indice)

    #breaking the data into training and testing data using K fold cross validation where k=15
    for k in range (1,16):
        X_test = [Sample_attributes_X[i] for i in indice[j:k*10]]
        Y_test = [Sample_labels_Y[i] for i in indice[j:k * 10]]
        if k<15:
            X_train=[Sample_attributes_X[i] for i in indice[k*10:]]
            Y_train = [Sample_labels_Y[i] for i in indice[k * 10:]]
        if j>0:
            X_train=[Sample_attributes_X[i] for i in indice[0:j]]
            Y_train = [Sample_labels_Y[i] for i in indice[0:j]]
        j=k*10
        A_X_train_shuffle=array(X_train)
        A_Y_train_shuffle=array(Y_train)
        A_X_test=array(X_test)
        A_Y_test=array(Y_test)
        Beta=[]

        #generating the value of all the coefficient and Y intercept in beta
        Beta.append(dot(dot(linalg.inv(dot(transpose(A_X_train_shuffle),A_X_train_shuffle)),transpose(A_X_train_shuffle)),A_Y_train_shuffle))

        #storing Beta as list for further calculations
        Beta=list(Beta[0])
        for i in range(0,len(Beta)):
            Beta[i]=list(Beta[i])

        for i in range(0, len(X_test)):
            #generating the value of all Y for evaluated Beta and X atributes in testing data
            Y_final = round(
                Beta[0][0] + Beta[1][0] * A_X_test[i][1] + Beta[2][0] * A_X_test[i][2] + Beta[3][0] * A_X_test[i][3] +
                Beta[4][0] * A_X_test[i][4])

            #comparing calculated Y value of testing data to actual data and generating correspondint true and false
            if Y_final == Y_test[i][0]:
                if Y_final == 1:
                    class_1["True_Positive"] += 1
                    class_2["True_Negative"] += 1
                    class_3["True_Negative"] += 1
                elif Y_final == 2:
                    class_2["True_Positive"] += 1
                    class_1["True_Negative"] += 1
                    class_3["True_Negative"] += 1
                elif Y_final == 3:
                    class_3["True_Positive"] += 1
                    class_1["True_Negative"] += 1
                    class_2["True_Negative"] += 1
            elif Y_final != Y_test[i][0]:
                if Y_test[i] == 1 and Y_final == 2:
                    class_2["False_Positive"] += 1
                    class_1["False_Negative"] += 1
                    class_3["True_Negative"] += 1

                elif Y_test[i][0] == 1 and Y_final == 3:
                    class_3["False_Positive"] += 1
                    class_1["False_Negative"] += 1
                    class_2["True_Negative"] += 1

                elif Y_test[i][0] == 2 and Y_final == 1:
                    class_1["False_Positive"] += 1
                    class_2["False_Negative"] += 1
                    class_3["True_Negative"] += 1

                elif Y_test[i][0] == 2 and Y_final == 3:
                    class_3["False_Positive"] += 1
                    class_2["False_Negative"] += 1
                    class_1["True_Negative"] += 1

                elif Y_test[i][0] == 3 and Y_final == 1:
                    class_1["False_Positive"] += 1
                    class_3["False_Negative"] += 1
                    class_2["True_Negative"] += 1

                elif Y_test[i][0] == 3 and Y_final == 2:
                    class_2["False_Positive"] += 1
                    class_3["False_Negative"] += 1
                    class_1["True_Negative"] += 1

        '''formula used for accuracy is (True Positive + True Negative)/(True Positive + True Negative + False Positive + False Negative)'''

        #checking the accuracy for Iris-Setsosa
        Accuracy_class_1.append(((float(class_1["True_Positive"]) + float(class_1["True_Negative"])) / (
            float(class_1["True_Positive"]) + float(class_1["True_Negative"]) + float(class_1["False_Positive"]) + float(class_1[
                "False_Negative"]))))

        #checking the accuracy for Iris-Versicolor
        Accuracy_class_2.append(((float(class_2["True_Positive"]) + float(class_2["True_Negative"])) / (
            float(class_2["True_Positive"]) + float(class_2["True_Negative"]) + float(class_2["False_Positive"]) + float(class_2[
                "False_Negative"]))))

        #checking the accuracy for Iris-Virginica
        Accuracy_class_3.append(((float(class_3["True_Positive"]) + float(class_3["True_Negative"])) / (
            float(class_3["True_Positive"]) + float(class_3["True_Negative"]) + float(class_3["False_Positive"]) + float(class_3[
                "False_Negative"]))))
        #calculating total values for ture positives, true negatives, false postives and false negatives
    Total_true_positive=float(class_1["True_Positive"])+float(class_2["True_Positive"])+float(class_3["True_Positive"])
    Total_true_negative=float(class_1["True_Negative"])+float(class_2["True_Negative"])+float(class_3["True_Negative"])
    Total_false_postive=float(class_1["False_Positive"])+float(class_2["False_Positive"])+float(class_3["False_Positive"])
    Total_false_negative=float(class_3["False_Negative"])+float(class_3["False_Negative"])+float(class_3["False_Negative"])

    #average accuracy for each class for 15 fold cross validation
    accuracy_class1=sum(Accuracy_class_1) / 15
    accuracy_class2=sum(Accuracy_class_2) / 15
    accuracy_class3=sum(Accuracy_class_3) / 15

    #Graph plot for accuracy v/s classes of Iris
    data = [go.Bar(x=['Iris-setosa', 'Iris-versicolor', 'Iris-Virginica'],
                y=[accuracy_class1*100,accuracy_class2*100,accuracy_class3*100],text = [accuracy_class1*100,accuracy_class2*100,accuracy_class3*100],
                       textposition = 'auto',
                                      marker = dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity = 0.6
        )]
    po.plot(data, filename='basic-bar.html')

    #printing the overall accuracy for Iris data set
    print "Overall Accuracy in percentage", ((Total_true_positive+Total_true_negative)/(Total_true_positive+Total_true_negative+Total_false_postive+Total_false_negative))*100

if __name__ == '__main__':
    print "\n\n*********Project 1: Linear regression on Iris data without using scikit libraries**********"
    print "**********************************submitted by*********************************************"
    print "****************************Student Name: Prachi Goel**************************************"
    print "******************************Student ID: 1001234789***************************************\n\n"
    data=reading_csv()
    training_classifier(data[0],data[1])

