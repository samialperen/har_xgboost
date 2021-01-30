# Author: Sami Alperen Akgun
# Email: sami.alperen.akgun@gmail.com

import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
import seaborn as sns #heatmap
from sklearn.model_selection import cross_val_score # apply cross validation
from sklearn.decomposition import PCA #feature extraction 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

 

def read_data(data_path, filename):
    """
        This function reads the data from data_path/filename
        WARNING: This function assumes that features of data
        is separated by whitespace in the file
        Input: data_path --> The full directory path of data
        filename --> name of the file (With extension)
        Output: 
                If X --> numpy array that contains feature values
                sample size = len(filename1) + len(filename2)
                size(X) --> sample size x feature number
                If Y --> numpy array that contains labels
                size(Y) --> (sample size,)
    """
    np_data = np.loadtxt(data_path + "/" + filename)
    return np_data

def class_breakdown(input_data):
    """
        This function prints the number and percentage of each class in
        a given input_data. This is just to see what kind of data we have.
        The best scenario is having a balanced data with equal number of 
        samples for each class.
    """
    # convert the numpy array into a dataframe
    df = pd.DataFrame(input_data)
    # group data by the class value and calculate the number of rows
    counts = df.groupby(0).size()
    counts = counts.values
	
    percent_list = []
    for i in range(len(counts)):
        percent_list.append((counts[i] / len(df)) * 100)
        #print('Class=%d, total=%d, percentage=%.3f' % (i+1, counts[i], percent_list[i])) 

    activity_list = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING",
                      "STANDING","LAYING"]
    
    figure_bar = plt.figure()
    bar_list = plt.bar(activity_list, percent_list,color='#557f2d', edgecolor='white')
    bar_list[0].set_color("r")
    bar_list[1].set_color("g")
    bar_list[2].set_color("b")
    bar_list[3].set_color("y")
    bar_list[4].set_color("c")
    bar_list[5].set_color("m")
    plt.title('Percentage of Activities in Given Data')
    plt.ylabel("Percentage (%)")
    plt.xlabel("Activity Names")   
 

def main():
    """
        This is the main function of this script
    """

    ##### Load data
    # If you run the code from pattern_recognition_assignment3 path, uncomment below
    data_dir = os.getcwd() + '/data/UCI_HAR_Dataset' 
    # If you run the code from code directory, uncomment below
    #data_path = os.getcwd() +  ".." / "data"/
    
    X_train = read_data(data_dir + "/train","X_train.txt")
    y_train = read_data(data_dir + "/train","y_train.txt")
    X_test = read_data(data_dir + "/test","X_test.txt")
    y_test = read_data(data_dir + "/test","y_test.txt")
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)

    ##### Check the number of samples in each class (balanced data?)
    class_breakdown(y_train)
    class_breakdown(y_test)

    ##### Plot Correlation Heat Map of the Features
    X_train_df = pd.read_csv(data_dir + "/train/X_train.txt", header=None, delim_whitespace=True)
    feat_f = open(data_dir + "/features.txt")
    feature_names = feat_f.readlines()
    feat_f.close()
    X_train_df.columns = feature_names # Change column names with feature names
    
    cmap=sns.diverging_palette(220,220,as_cmap=True)
    fig1 = plt.figure()
    ax=sns.heatmap(X_train_df.corr(),cmap=cmap)
    plt.title("Correlation Heat Map of Features")
    plt.xlabel("features")
    plt.ylabel("features")

    ##### Shuffle Training Data for better performance
    x1_size, x2_size = X_train.shape
    combined_data = np.concatenate((X_train,y_train.reshape(x1_size,1)),axis=1)
    np.random.shuffle(combined_data) #this function shuffles

    X_train_shuf = combined_data[:,0:x2_size]
    y_train_shuf = combined_data[:,x2_size:]
    y_train_shuf = y_train_shuf.reshape(y_train_shuf.shape[0])
    print("X_train_shuf shape: ", X_train_shuf.shape)
    print("y_train_shuf shape: ", y_train_shuf.shape)
 
    ##### Feature Extraction - Principal Component Analysis (PCA)
    # Standardize training set to mean zero variance 1
    X_train_stand = StandardScaler().fit_transform(X_train_shuf) 
    X_test_stand = StandardScaler().fit_transform(X_test) 

    pca = PCA(n_components=X_train_shuf.shape[1]) # use all features
    pca.fit(X_train_stand)
    
    """
        Principal component plotting part is taken from the repo below:
        https://github.com/nilesh-patil/human-activity-recognition-smartphone-sensors/blob/
        master/code/002.PCA.ipynb
    """
    components = {'pc'+str(id):pca.explained_variance_ratio_[id] for id in range(X_train_stand.shape[1])}
    principle_components = pd.DataFrame.from_dict(components,orient='index')
    principle_components.columns=['varExplained']
    principle_components.sort_values(by='varExplained', ascending=False,inplace=True)
    data_plot = principle_components
    data_plot['component'] = data_plot.index
    data_plot['totalVarExplained'] = data_plot.varExplained.cumsum(0)

    # Plot first 20 principal component
    n = 20
    plt.figure()
    sns.barplot(x='component',y='varExplained',data=data_plot.head(n),
                palette=sns.color_palette("Blues_r",n_colors=n))
    plt.plot(range(n),data_plot.totalVarExplained[:n],'--')
    plt.xlabel('Principle component')
    plt.ylabel('Total fraction of variance explained')
    plt.title('Total variance explained vs number of component')
    
    # Take all the components such that 95% of the variance is retained
    pca2 = PCA(0.95)
    pca2.fit(X_train_stand)
    print("Number of pca components for 95% variance: ", pca2.n_components_)

    X_train_pca = pca2.transform(X_train_stand)
    X_test_pca = pca2.transform(X_test_stand)

    learning_rate = 0.6
    estimator_number = 800
    depth = 2
    clf_xgbo = XGBClassifier(learning_rate=learning_rate, n_estimators=estimator_number,max_depth=depth,
                             objective="reg:logistic")
    clf_xgbo.fit(X_train_pca,y_train_shuf)

    ##### Parameter tuning
    #prediction_xgbo = clf_xgbo.predict(X_test_pca)
    #accuracy_xgbo = accuracy_score(y_test, prediction_xgbo)*100
    #print('Extreme Gradient Boosting Accuracy: ', accuracy_xgbo)

    
    cv_scores = []
    for i in range(10): #10 times 
        cv_scores.append(cross_val_score(clf_xgbo, X_test, y_test, cv=10)*100) #10-fold cv


    print("10-times-10-fold CV")
    print("Best Accuracy: ", np.max(cv_scores))
    print("Mean Accuracy: ", np.mean(cv_scores))
    print("Var Accuracy: ", np.var(cv_scores))

    prediction_xgbo = clf_xgbo.predict(X_test_pca)
    accuracy_xgbo = accuracy_score(y_test, prediction_xgbo)*100
    print('Extreme Gradient Boosting Accuracy: ', accuracy_xgbo)

    label = [1,2,3,4,5,6]
    conf_matrix = confusion_matrix(y_test,prediction_xgbo,labels=label)
    print("Confusion Matrix")
    print(conf_matrix)





    plt.show(block=False) # show all the figures at once
    plt.waitforbuttonpress(1)
    input("Please press any key to close all figures.")
    plt.close("all")


if __name__ == "__main__": main()
