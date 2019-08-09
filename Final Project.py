#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Helper Functions
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

def draw_heatmap_dt(acc, acc_desc, depth_list):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(acc, annot=True, fmt='.3f', 
                     yticklabels=depth_list, xticklabels=[])
    ax.collections[0].colorbar.set_label("accuracy")
    ax.set(ylabel='depth')
    plt.title(acc_desc + ' w.r.t depth')
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.show()
    
def draw_heatmap_svm(acc, acc_desc, C_list):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(acc, annot=True, fmt='.3f', yticklabels=C_list, xticklabels=[])
    ax.collections[0].colorbar.set_label("accuracy")
    ax.set(ylabel='$C$')
    plt.title(acc_desc + ' w.r.t $C$')
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.show()
    
def draw_heatmap_rfc(acc, acc_desc):
    plt.figure(figsize = (2,4))
    ax = sns.heatmap(acc, annot=True, fmt='.3f', 
                     yticklabels=[], xticklabels=[])
    ax.collections[0].colorbar.set_label("accuracy")
    plt.title(acc_desc)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.show()
    
def testacc(prediction, actual):
    correct = 0
    for i in range(len(prediction)):
        if prediction[i] == actual[i]:
            correct += 1
    print("Test accuracy = ", end =" ")
    print(correct / len(prediction))


# In[ ]:


#Import and format data set 1
X_and_Y = np.loadtxt('dota2data.csv', delimiter=",")
np.random.shuffle(X_and_Y)
X = X_and_Y[:, 1:]
Y = X_and_Y[:, 0]
print(X.shape, Y.shape)

X_train_val1 = X[:int(0.8*len(X))]
X_test1      = X[int(0.8*len(X)):]
Y_train_val1 = Y[:int(0.8*len(Y))]
Y_test1      = Y[int(0.8*len(Y)):]
print(X_train_val1.shape, X_test1.shape, 
      Y_train_val1.shape, Y_test1.shape)

X_train_val2 = X[:int(0.5*len(X))]
X_test2      = X[int(0.5*len(X)):]
Y_train_val2 = Y[:int(0.5*len(Y))]
Y_test2      = Y[int(0.5*len(Y)):]
print(X_train_val2.shape, X_test2.shape, 
      Y_train_val2.shape, Y_test2.shape)

X_train_val3 = X[:int(0.2*len(X))]
X_test3      = X[int(0.2*len(X)):]    
Y_train_val3 = Y[:int(0.2*len(Y))]
Y_test3      = Y[int(0.2*len(Y)):]
print(X_train_val3.shape, X_test3.shape, 
      Y_train_val3.shape, Y_test3.shape)


# In[ ]:


#DT for data set 1
param = {'max_depth': [1, 2, 3, 4, 5], 
         'criterion': ['entropy']}
depthlist = np.matrix([1, 2, 3, 4, 5])

tree1 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree1.fit(X_train_val1, Y_train_val1)

tree2 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree2.fit(X_train_val2, Y_train_val2)

tree3 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree3.fit(X_train_val3, Y_train_val3)


# In[ ]:


#Calculating stats for DT using data set 1
acc1 = np.matrix([tree1.cv_results_["split0_test_score"], 
                tree1.cv_results_["split1_test_score"], 
                tree1.cv_results_["split2_test_score"], 
                tree1.cv_results_["split3_test_score"], 
                tree1.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc1, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree1.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree1.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree1.cv_results_["mean_test_score"])
testacc(tree1.predict(X_test1), Y_test1)

acc2 = np.matrix([tree2.cv_results_["split0_test_score"], 
                tree2.cv_results_["split1_test_score"], 
                tree2.cv_results_["split2_test_score"], 
                tree2.cv_results_["split3_test_score"], 
                tree2.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc2, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree2.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree2.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree2.cv_results_["mean_test_score"])
testacc(tree2.predict(X_test2), Y_test2)

acc3 = np.matrix([tree3.cv_results_["split0_test_score"], 
                tree3.cv_results_["split1_test_score"], 
                tree3.cv_results_["split2_test_score"], 
                tree3.cv_results_["split3_test_score"], 
                tree3.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc3, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree3.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree3.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree3.cv_results_["mean_test_score"])
testacc(tree3.predict(X_test3), Y_test3)


# In[ ]:


#SVM for data set 1
classifier = LinearSVC()
C_list = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
parameters = {'C':C_list}

clf1 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf1.fit(X_train_val1, np.ravel(Y_train_val1))

clf2 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf2.fit(X_train_val2, np.ravel(Y_train_val2))

clf3 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf3.fit(X_train_val3, np.ravel(Y_train_val3))


# In[ ]:


#Calculating stats for SVM using data set 1
train_acc1 = np.array(np.matrix(
    clf1.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc1, 'train accuracy', C_list)
val_acc1 = np.array(np.matrix(
    clf1.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc1, 'val accuracy', C_list)
testacc(clf1.predict(X_test1), Y_test1)
print("Optimal C =", end =" ")
print(clf1.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf1.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf1.cv_results_["mean_test_score"])

train_acc2 = np.array(np.matrix(
    clf2.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc2, 'train accuracy', C_list)
val_acc2 = np.array(np.matrix(
    clf2.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc2, 'val accuracy', C_list)
testacc(clf2.predict(X_test2), Y_test2)
print("Optimal C =", end =" ")
print(clf2.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf2.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf2.cv_results_["mean_test_score"])

train_acc3 = np.array(np.matrix(
    clf3.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc3, 'train accuracy', C_list)
val_acc3 = np.array(np.matrix(
    clf3.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc3, 'val accuracy', C_list)
testacc(clf3.predict(X_test3), Y_test3)
print("Optimal C =", end =" ")
print(clf3.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf3.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf3.cv_results_["mean_test_score"])


# In[ ]:


#RFC for data set 1
n_list = np.array([5, 8, 10, 12, 15])
md_list = np.array([1, 2, 3, 4, 5])
parameters = {'n_estimators':n_list, 'max_depth':md_list}

rfc1 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc1.fit(X_train_val1, np.ravel(Y_train_val1))

rfc2 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc2.fit(X_train_val2, np.ravel(Y_train_val2))

rfc3 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc3.fit(X_train_val3, np.ravel(Y_train_val3))


# In[ ]:


#Calculating stats for RFC using data set 1
bestdepth1 = rfc1.best_params_["max_depth"]
bestn1 = rfc1.best_params_["n_estimators"]
index1 = list(md_list).index(bestdepth1) * 5 - 1            + list(n_list).index(bestn1)
acc1 = np.matrix([rfc1.cv_results_["split0_test_score"][index1], 
                rfc1.cv_results_["split1_test_score"][index1], 
                rfc1.cv_results_["split2_test_score"][index1], 
                rfc1.cv_results_["split3_test_score"][index1], 
                rfc1.cv_results_["split4_test_score"][index1]])
draw_heatmap_rfc(acc1, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc1.best_params_["max_depth"])
print("Optimal n estimators", end =" ")
print(rfc1.best_params_["n_estimators"])
print("Average Train Error =", end =" ")
print(rfc1.cv_results_["mean_train_score"][index1])
print("Average Test Error =", end =" ")
print(rfc1.cv_results_["mean_test_score"][index1])
testacc(rfc1.predict(X_test1), Y_test1)

bestdepth2 = rfc2.best_params_["max_depth"]
bestn2 = rfc2.best_params_["n_estimators"]
index2 = list(md_list).index(bestdepth2) * 5 - 1            + list(n_list).index(bestn2)
acc2 = np.matrix([rfc2.cv_results_["split0_test_score"][index2], 
                rfc2.cv_results_["split1_test_score"][index2], 
                rfc2.cv_results_["split2_test_score"][index2], 
                rfc2.cv_results_["split3_test_score"][index2], 
                rfc2.cv_results_["split4_test_score"][index2]])
draw_heatmap_rfc(acc2, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc2.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(rfc2.cv_results_["mean_train_score"][index2])
print("Average Test Error =", end =" ")
print(rfc2.cv_results_["mean_test_score"][index2])
testacc(rfc2.predict(X_test2), Y_test2)

bestdepth3 = rfc3.best_params_["max_depth"]
bestn3 = rfc3.best_params_["n_estimators"]
index3 = list(md_list).index(bestdepth3) * 5 - 1            + list(n_list).index(bestn3)
acc3 = np.matrix([rfc3.cv_results_["split0_test_score"][index3], 
                rfc3.cv_results_["split1_test_score"][index3], 
                rfc3.cv_results_["split2_test_score"][index3], 
                rfc3.cv_results_["split3_test_score"][index3], 
                rfc3.cv_results_["split4_test_score"][index3]])
draw_heatmap_rfc(acc3, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc3.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(rfc3.cv_results_["mean_train_score"][index3])
print("Average Test Error =", end =" ")
print(rfc3.cv_results_["mean_test_score"][index3])
testacc(rfc3.predict(X_test3), Y_test3)


# In[ ]:


#Import and format data set 2
X_and_Y = np.loadtxt('fmt-connect-4.data', delimiter=",")
np.random.shuffle(X_and_Y)
X = X_and_Y[:, 0:-1]
Y = X_and_Y[:, -1]
print(X.shape, Y.shape)

X_train_val1 = X[:int(0.8*len(X))]
X_test1      = X[int(0.8*len(X)):]   
Y_train_val1 = Y[:int(0.8*len(Y))]
Y_test1      = Y[int(0.8*len(Y)):]
print(X_train_val1.shape, X_test1.shape, 
      Y_train_val1.shape, Y_test1.shape)

X_train_val2 = X[:int(0.5*len(X))]
X_test2      = X[int(0.5*len(X)):]    
Y_train_val2 = Y[:int(0.5*len(Y))]
Y_test2      = Y[int(0.5*len(Y)):]
print(X_train_val2.shape, X_test2.shape, 
      Y_train_val2.shape, Y_test2.shape)

X_train_val3 = X[:int(0.2*len(X))]
X_test3      = X[int(0.2*len(X)):]    
Y_train_val3 = Y[:int(0.2*len(Y))]
Y_test3      = Y[int(0.2*len(Y)):]
print(X_train_val3.shape, X_test3.shape, 
      Y_train_val3.shape, Y_test3.shape)


# In[ ]:


#DT for data set 2
param = {'max_depth': [1, 2, 3, 4, 5], 
         'criterion': ['entropy']}
depthlist = np.matrix([1, 2, 3, 4, 5])

tree1 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree1.fit(X_train_val1, Y_train_val1)

tree2 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree2.fit(X_train_val2, Y_train_val2)

tree3 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree3.fit(X_train_val3, Y_train_val3)


# In[ ]:


#Calculating stats for DT using data set 2
acc1 = np.matrix([tree1.cv_results_["split0_test_score"], 
                tree1.cv_results_["split1_test_score"], 
                tree1.cv_results_["split2_test_score"], 
                tree1.cv_results_["split3_test_score"], 
                tree1.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc1, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree1.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree1.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree1.cv_results_["mean_test_score"])
testacc(tree1.predict(X_test1), Y_test1)

acc2 = np.matrix([tree2.cv_results_["split0_test_score"], 
                tree2.cv_results_["split1_test_score"], 
                tree2.cv_results_["split2_test_score"], 
                tree2.cv_results_["split3_test_score"], 
                tree2.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc2, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree2.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree2.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree2.cv_results_["mean_test_score"])
testacc(tree2.predict(X_test2), Y_test2)

acc3 = np.matrix([tree3.cv_results_["split0_test_score"], 
                tree3.cv_results_["split1_test_score"], 
                tree3.cv_results_["split2_test_score"], 
                tree3.cv_results_["split3_test_score"], 
                tree3.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc3, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree3.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree3.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree3.cv_results_["mean_test_score"])
testacc(tree3.predict(X_test3), Y_test3)


# In[ ]:


#SVM for data set 2
classifier = LinearSVC()
C_list = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
parameters = {'C':C_list}

clf1 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf1.fit(X_train_val1, np.ravel(Y_train_val1))

clf2 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf2.fit(X_train_val2, np.ravel(Y_train_val2))

clf3 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf3.fit(X_train_val3, np.ravel(Y_train_val3))


# In[ ]:


#Calculating stats for SVM using data set 2
train_acc1 = np.array(np.matrix(
    clf1.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc1, 'train accuracy', C_list)
val_acc1 = np.array(np.matrix(
    clf1.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc1, 'val accuracy', C_list)
testacc(clf1.predict(X_test1), Y_test1)
print("Optimal C =", end =" ")
print(clf1.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf1.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf1.cv_results_["mean_test_score"])

train_acc2 = np.array(np.matrix(
    clf2.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc2, 'train accuracy', C_list)
val_acc2 = np.array(np.matrix(
    clf2.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc2, 'val accuracy', C_list)
testacc(clf2.predict(X_test2), Y_test2)
print("Optimal C =", end =" ")
print(clf2.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf2.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf2.cv_results_["mean_test_score"])

train_acc3 = np.array(np.matrix(
    clf3.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc3, 'train accuracy', C_list)
val_acc3 = np.array(np.matrix(
    clf3.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc3, 'val accuracy', C_list)
testacc(clf3.predict(X_test3), Y_test3)
print("Optimal C =", end =" ")
print(clf3.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf3.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf3.cv_results_["mean_test_score"])


# In[ ]:


#RFC for data set 2
n_list = np.array([5, 8, 10, 12, 15])
md_list = np.array([1, 2, 3, 4, 5])
parameters = {'n_estimators':n_list, 'max_depth':md_list}

rfc1 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc1.fit(X_train_val1, np.ravel(Y_train_val1))

rfc2 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc2.fit(X_train_val2, np.ravel(Y_train_val2))

rfc3 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc3.fit(X_train_val3, np.ravel(Y_train_val3))


# In[ ]:


#Calculating stats for RFC using data set 2
bestdepth1 = rfc1.best_params_["max_depth"]
bestn1 = rfc1.best_params_["n_estimators"]
index1 = list(md_list).index(bestdepth1) * 5 - 1            + list(n_list).index(bestn1)
acc1 = np.matrix([rfc1.cv_results_["split0_test_score"][index1], 
                rfc1.cv_results_["split1_test_score"][index1], 
                rfc1.cv_results_["split2_test_score"][index1], 
                rfc1.cv_results_["split3_test_score"][index1], 
                rfc1.cv_results_["split4_test_score"][index1]])
draw_heatmap_rfc(acc1, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc1.best_params_["max_depth"])
print("Optimal n estimators", end =" ")
print(rfc1.best_params_["n_estimators"])
print("Average Train Error =", end =" ")
print(rfc1.cv_results_["mean_train_score"][index1])
print("Average Test Error =", end =" ")
print(rfc1.cv_results_["mean_test_score"][index1])
testacc(rfc1.predict(X_test1), Y_test1)

bestdepth2 = rfc2.best_params_["max_depth"]
bestn2 = rfc2.best_params_["n_estimators"]
index2 = list(md_list).index(bestdepth2) * 5 - 1            + list(n_list).index(bestn2)
acc2 = np.matrix([rfc2.cv_results_["split0_test_score"][index2], 
                rfc2.cv_results_["split1_test_score"][index2], 
                rfc2.cv_results_["split2_test_score"][index2], 
                rfc2.cv_results_["split3_test_score"][index2], 
                rfc2.cv_results_["split4_test_score"][index2]])
draw_heatmap_rfc(acc2, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc2.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(rfc2.cv_results_["mean_train_score"][index2])
print("Average Test Error =", end =" ")
print(rfc2.cv_results_["mean_test_score"][index2])
testacc(rfc2.predict(X_test2), Y_test2)

bestdepth3 = rfc3.best_params_["max_depth"]
bestn3 = rfc3.best_params_["n_estimators"]
index3 = list(md_list).index(bestdepth3) * 5 - 1            + list(n_list).index(bestn3)
acc3 = np.matrix([rfc3.cv_results_["split0_test_score"][index3], 
                rfc3.cv_results_["split1_test_score"][index3], 
                rfc3.cv_results_["split2_test_score"][index3], 
                rfc3.cv_results_["split3_test_score"][index3], 
                rfc3.cv_results_["split4_test_score"][index3]])
draw_heatmap_rfc(acc3, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc3.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(rfc3.cv_results_["mean_train_score"][index3])
print("Average Test Error =", end =" ")
print(rfc3.cv_results_["mean_test_score"][index3])
testacc(rfc3.predict(X_test3), Y_test3)


# In[ ]:


#Import and format data set 3
X_and_Y = np.loadtxt('formatted_poker.txt', delimiter=",")
np.random.shuffle(X_and_Y)
#Using all 5 cards
X = X_and_Y[:, 0:-1]
#Using 4 cards
#X = X_and_Y[:, 0:-3]
#Using 3 cards
#X = X_and_Y[:, 0:-5]
#Using 2 cards
#X = X_and_Y[:, 0:-7]
#Using 1 card
#X = X_and_Y[:, 0:-9]
Y = X_and_Y[:, -1]
print(X.shape, Y.shape)

X_train_val1 = X[:int(0.8*len(X))]
X_test1      = X[int(0.8*len(X)):]   
Y_train_val1 = Y[:int(0.8*len(Y))]
Y_test1      = Y[int(0.8*len(Y)):]
print(X_train_val1.shape, X_test1.shape, 
      Y_train_val1.shape, Y_test1.shape)

X_train_val2 = X[:int(0.5*len(X))]
X_test2      = X[int(0.5*len(X)):]
Y_train_val2 = Y[:int(0.5*len(Y))]
Y_test2      = Y[int(0.5*len(Y)):]
print(X_train_val2.shape, X_test2.shape, 
      Y_train_val2.shape, Y_test2.shape)

X_train_val3 = X[:int(0.2*len(X))]
X_test3      = X[int(0.2*len(X)):]
Y_train_val3 = Y[:int(0.2*len(Y))]
Y_test3      = Y[int(0.2*len(Y)):]
print(X_train_val3.shape, X_test3.shape, 
      Y_train_val3.shape, Y_test3.shape)


# In[ ]:


#DT for data set 3
param = {'max_depth': [1, 2, 3, 4, 5], 
         'criterion': ['entropy']}
depthlist = np.matrix([1, 2, 3, 4, 5])

tree1 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree1.fit(X_train_val1, Y_train_val1)

tree2 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree2.fit(X_train_val2, Y_train_val2)

tree3 = GridSearchCV(DecisionTreeClassifier(), param, cv=5, 
                     return_train_score=True)
tree3.fit(X_train_val3, Y_train_val3)


# In[ ]:


#Calculating stats for DT using data set 3
acc1 = np.matrix([tree1.cv_results_["split0_test_score"], 
                tree1.cv_results_["split1_test_score"], 
                tree1.cv_results_["split2_test_score"], 
                tree1.cv_results_["split3_test_score"], 
                tree1.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc1, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree1.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree1.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree1.cv_results_["mean_test_score"])
testacc(tree1.predict(X_test1), Y_test1)

acc2 = np.matrix([tree2.cv_results_["split0_test_score"], 
                tree2.cv_results_["split1_test_score"], 
                tree2.cv_results_["split2_test_score"], 
                tree2.cv_results_["split3_test_score"], 
                tree2.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc2, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree2.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree2.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree2.cv_results_["mean_test_score"])
testacc(tree2.predict(X_test2), Y_test2)

acc3 = np.matrix([tree3.cv_results_["split0_test_score"], 
                tree3.cv_results_["split1_test_score"], 
                tree3.cv_results_["split2_test_score"], 
                tree3.cv_results_["split3_test_score"], 
                tree3.cv_results_["split4_test_score"]])
draw_heatmap_dt(acc3, "Validation accuracy per fold", depthlist)
print("Optimal maximum depth =", end =" ") 
print(tree3.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(tree3.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(tree3.cv_results_["mean_test_score"])
testacc(tree3.predict(X_test3), Y_test3)


# In[ ]:


#SVM for data set 3
classifier = LinearSVC()
C_list = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1])
parameters = {'C':C_list}

clf1 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf1.fit(X_train_val1, np.ravel(Y_train_val1))

clf2 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf2.fit(X_train_val2, np.ravel(Y_train_val2))

clf3 = GridSearchCV(classifier, parameters, cv=5,
                   return_train_score=True)
clf3.fit(X_train_val3, np.ravel(Y_train_val3))


# In[ ]:


#Calculating stats for SVM using data set 3
train_acc1 = np.array(np.matrix(
    clf1.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc1, 'train accuracy', C_list)
val_acc1 = np.array(np.matrix(
    clf1.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc1, 'val accuracy', C_list)
testacc(clf1.predict(X_test1), Y_test1)
print("Optimal C =", end =" ")
print(clf1.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf1.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf1.cv_results_["mean_test_score"])

train_acc2 = np.array(np.matrix(
    clf2.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc2, 'train accuracy', C_list)
val_acc2 = np.array(np.matrix(
    clf2.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc2, 'val accuracy', C_list)
testacc(clf2.predict(X_test2), Y_test2)
print("Optimal C =", end =" ")
print(clf2.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf2.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf2.cv_results_["mean_test_score"])

train_acc3 = np.array(np.matrix(
    clf3.cv_results_['mean_train_score']).transpose())
draw_heatmap_svm(train_acc3, 'train accuracy', C_list)
val_acc3 = np.array(np.matrix(
    clf3.cv_results_['mean_test_score']).transpose())
draw_heatmap_svm(val_acc3, 'val accuracy', C_list)
testacc(clf3.predict(X_test3), Y_test3)
print("Optimal C =", end =" ")
print(clf3.best_params_["C"])
print("Average Train Error =", end =" ")
print(clf3.cv_results_["mean_train_score"])
print("Average Test Error =", end =" ")
print(clf3.cv_results_["mean_test_score"])


# In[ ]:


#RFC for data set 3
n_list = np.array([5, 8, 10, 12, 15])
md_list = np.array([1, 2, 3, 4, 5])
parameters = {'n_estimators':n_list, 'max_depth':md_list}

rfc1 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc1.fit(X_train_val1, np.ravel(Y_train_val1))

rfc2 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc2.fit(X_train_val2, np.ravel(Y_train_val2))

rfc3 = GridSearchCV(RandomForestClassifier(), parameters, cv=5,
                    return_train_score=True)
rfc3.fit(X_train_val3, np.ravel(Y_train_val3))


# In[ ]:


#Calculating stats for RFC using data set 3
bestdepth1 = rfc1.best_params_["max_depth"]
bestn1 = rfc1.best_params_["n_estimators"]
index1 = list(md_list).index(bestdepth1) * 5 - 1            + list(n_list).index(bestn1)
acc1 = np.matrix([rfc1.cv_results_["split0_test_score"][index1], 
                rfc1.cv_results_["split1_test_score"][index1], 
                rfc1.cv_results_["split2_test_score"][index1], 
                rfc1.cv_results_["split3_test_score"][index1], 
                rfc1.cv_results_["split4_test_score"][index1]])
draw_heatmap_rfc(acc1, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc1.best_params_["max_depth"])
print("Optimal n estimators", end =" ")
print(rfc1.best_params_["n_estimators"])
print("Average Train Error =", end =" ")
print(rfc1.cv_results_["mean_train_score"][index1])
print("Average Test Error =", end =" ")
print(rfc1.cv_results_["mean_test_score"][index1])
testacc(rfc1.predict(X_test1), Y_test1)

bestdepth2 = rfc2.best_params_["max_depth"]
bestn2 = rfc2.best_params_["n_estimators"]
index2 = list(md_list).index(bestdepth2) * 5 - 1            + list(n_list).index(bestn2)
acc2 = np.matrix([rfc2.cv_results_["split0_test_score"][index2], 
                rfc2.cv_results_["split1_test_score"][index2], 
                rfc2.cv_results_["split2_test_score"][index2], 
                rfc2.cv_results_["split3_test_score"][index2], 
                rfc2.cv_results_["split4_test_score"][index2]])
draw_heatmap_rfc(acc2, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc2.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(rfc2.cv_results_["mean_train_score"][index2])
print("Average Test Error =", end =" ")
print(rfc2.cv_results_["mean_test_score"][index2])
testacc(rfc2.predict(X_test2), Y_test2)

bestdepth3 = rfc3.best_params_["max_depth"]
bestn3 = rfc3.best_params_["n_estimators"]
index3 = list(md_list).index(bestdepth3) * 5 - 1            + list(n_list).index(bestn3)
acc3 = np.matrix([rfc3.cv_results_["split0_test_score"][index3], 
                rfc3.cv_results_["split1_test_score"][index3], 
                rfc3.cv_results_["split2_test_score"][index3], 
                rfc3.cv_results_["split3_test_score"][index3], 
                rfc3.cv_results_["split4_test_score"][index3]])
draw_heatmap_rfc(acc3, "Validation accuracy per fold")
print("Optimal maximum depth =", end =" ") 
print(rfc3.best_params_["max_depth"])
print("Average Train Error =", end =" ")
print(rfc3.cv_results_["mean_train_score"][index3])
print("Average Test Error =", end =" ")
print(rfc3.cv_results_["mean_test_score"][index3])
testacc(rfc3.predict(X_test3), Y_test3)

