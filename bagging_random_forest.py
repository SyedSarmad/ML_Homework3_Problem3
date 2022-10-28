#-------------------------------------------------------------------------
# AUTHOR: Syed Sarmad
# FILENAME: bagging_random_forest.py
# SPECIFICATION: Use thre different types of classifiers to clasify optical reading data
# FOR: CS 4210- Assignment #3
# TIME SPENT: 5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv


dbTraining = []
dbTest = []
X_training = []
y_training = []
classVotes = [] #this array will be used to count the votes of each classifier

classifier_1_correct = 0
classifier_1_incorrect = 0

classifier_2_correct = 0
classifier_2_incorrect = 0

classifier_3_correct = 0
classifier_3_incorrect = 0

#reading the training data from a csv file and populate dbTraining
#--> add your Python code here

#reading the training data in a csv file
with open('/Users/sarmad/Desktop/ML HW 3 Problem 3/optdigits.tra', 'r') as csvfile:
   reader = csv.reader(csvfile)
   for i, row in enumerate(reader):
         ##if i > 0: #skipping the header
         dbTraining.append (row)



#reading the test data from a csv file and populate dbTest
#--> add your Python code here

#reading the test data in a csv file
with open('/Users/sarmad/Desktop/ML HW 3 Problem 3/optdigits.tes', 'r') as csvfile:
   reader = csv.reader(csvfile)
   for i, row in enumerate(reader):
         ##if i > 0: #skipping the header
         dbTest.append (row)

#inititalizing the class votes for each test sample. Example: classVotes.append([0,0,0,0,0,0,0,0,0,0])
#--> add your Python code here


rows_in_dbtest = len(dbTest)

classVotes = [[0] * 10 for i in range(rows_in_dbtest)]


print("Started my base and ensemble classifier ...")

for k in range(20): #we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample
   bootstrapSample = resample(dbTraining, n_samples=len(dbTraining), replace=True)
   
   #populate the values of X_training and y_training by using the bootstrapSample

   #--> add your Python code here
   #################################################################################
   for val in bootstrapSample:
      #X_training has everything but the last element 0-63 (64 elements each row)
      X_training.append(val[:-1])
      #Y_training has the list of the last elements (last element each row)
      y_training.append(val[-1])
  ################################################################################

  #fitting the decision tree to the data

   clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=None) #we will use a single decision tree without pruning it
   clf = clf.fit(X_training, y_training)

   for i, testSample in enumerate(dbTest):

      #make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
      # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
      # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
      # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
      # this array will consolidate the votes of all classifier for all test samples
      #--> add your Python code here

      ################################################################################
      #making a prediction based on the test sample
      y_predict = int (clf.predict([testSample[:-1]])[0])
      #incrementing class votes for this selection
      classVotes[i][y_predict] = classVotes[i][y_predict] + 1
      #getting the actual result from the sample; its the last spot since thats where the y's are located 
      y_test = int(testSample[-1])
      ################################################################################

      
      if k == 0: #for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
      #--> add your Python code here
         if y_test == y_predict:
            classifier_1_correct += 1
         else:
            classifier_1_incorrect += 1

   if k == 0: #for only the first base classifier, print its accuracy here
      #--> add your Python code here
      accuracy = classifier_1_correct / (classifier_1_correct + classifier_1_incorrect)

      print("Finished my base classifier (fast but relatively low accuracy) ...")
      print("My base classifier accuracy: " + str(accuracy))
      print("")

   #now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
   #--> add your Python code here

   #################################################################################
   for i, testSample in enumerate(dbTest):

      #getting the majority vote for that class
      majority = int(classVotes[i].index(max(classVotes[i])))
      #getting the actual value
      y_test = int(testSample[-1])
      if majority == y_test:
         classifier_2_correct += 1
      else:
         classifier_2_incorrect += 1
   #################################################################################

   #printing the ensemble accuracy here
   accuracy = classifier_2_correct / (classifier_1_correct + classifier_2_incorrect)  

   print("Finished my ensemble classifier (slow but higher accuracy) ...")
   print("My ensemble accuracy: " + str(accuracy))
   print("")

   print("Started Random Forest algorithm ...")

   #Create a Random Forest Classifier
   clf = RandomForestClassifier(n_estimators=20) #this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before

   #Fit Random Forest to the training data
   clf.fit(X_training,y_training)

   #make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
   #--> add your Python code here

   #################################################################################
   for i, testSample in enumerate(dbTest):
      random_forest_prediction = int(clf.predict([testSample[:-1]]))
      y_test = int(testSample[-1])
   #################################################################################

   #compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
   #--> add your Python code here

   #################################################################################
   if random_forest_prediction == y_test:
      classifier_3_correct += 1
   else:
      classifier_3_incorrect += 1

   accuracy = classifier_3_correct / (classifier_3_correct + classifier_3_incorrect)
   #################################################################################
   
   #printing Random Forest accuracy here
   print("Random Forest accuracy: " + str(accuracy))

print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
