from Perceptron import Perceptron
from LinearKernel import LinearKernel
from RegularizedLogisticRegression import RegularizedLogisticRegression
from SVM import SVM
from SVM_AUC import SVM_AUC
from SVMMulticlass import SVMMulticlass
from SVMMulticlassAUC import SVMMulticlassAUC
from sklearn.utils import shuffle
import pandas as pd


def LoadData(path):
    dataSet = pd.read_csv(filepath_or_buffer=path, header=None)
    dataSet = dataSet.fillna(0)
    dataSet = dataSet.drop_duplicates()
    dataSet = shuffle(dataSet)
    return dataSet


if __name__ == '__main__':
 
    perceptronDataDataSet = LoadData('./dataset/perceptronData.csv')
    twoSpiralsDataSet = LoadData('./dataset/twoSpirals.csv')
    spambaseDataSet = LoadData( './dataset/spambase.csv')       
    diabetesDataSet = LoadData( './dataset/diabetes.csv')
    breastCancerDataSet = LoadData('./dataset/breastcancer.csv')
    wineDataSet = LoadData('./dataset/wine.csv')
    
    
    
    
#1

    print("1.2 Perceptron - PerceptronData")
    perceptronDataPerceptron = Perceptron(perceptronDataDataSet)
    perceptronDataPerceptron.validate()

    for i in (1, 5, 10, 20):
        print("1.3 Linear Kernel - Two Spiral")
        twoSpiralLinearPerceptron = LinearKernel(twoSpiralsDataSet, i)
        twoSpiralLinearPerceptron.validate()
    
    
    print ('--------------------------------------------------------')

    for i in (1, 5, 10, 20):
        print("1.3 Gaussian Kernel - Two Spiral")
        twoSpiralGuassianPerceptron = LinearKernel(twoSpiralsDataSet, i,False)
        twoSpiralGuassianPerceptron.validate()
   
    print ('--------------------------------------------------------')







#2



    print("2 Logistic Regression - Spambase")
    spambaseLogisticRegression = RegularizedLogisticRegression(
        spambaseDataSet, 'Spambase', 0.75, 0.00001)
    spambaseLogisticRegression.validate()

    print ('--------------------------------------------------------')

    print("2 Logistic Regression - Diabetes")
    diabetesLogisticRegression = RegularizedLogisticRegression(
        diabetesDataSet, 'Diabetes', 0.1, 0.0000001)
    diabetesLogisticRegression.validate()

    print ('--------------------------------------------------------')

    print("2 Logistic Regression - Breast Cancer")
    breastCancerLogisticRegression = RegularizedLogisticRegression(
        breastCancerDataSet, 'Cancer', 0.75, 0.00001)
    breastCancerLogisticRegression.validate()

    print ('--------------------------------------------------------')




#3

    print("3 SVM Linear - Diabetes")
    diabetesSVM = SVM(diabetesDataSet)
    diabetesSVM.validate()

    print ('--------------------------------------------------------')

    print("3 SVM RBF - Diabetes")
    diabetesSVM = SVM(diabetesDataSet, False)
    diabetesSVM.validate()

    print ('--------------------------------------------------------')


    print("3 SVM Linear AUC - Diabetes")
    diabetesSVM = SVM_AUC(diabetesDataSet)
    diabetesSVM.validate()

    print ('--------------------------------------------------------')

    print("3 SVM RBF AUC - Diabetes")
    diabetesSVMAUC = SVM_AUC(diabetesDataSet, False)
    diabetesSVMAUC.validate()
    
    print ('--------------------------------------------------------')

    
    
    
    
    
    print("3 SVM Linear - Breast Cancer")
    breastCancerSVM = SVM(breastCancerDataSet)
    breastCancerSVM.validate()
    print ('--------------------------------------------------------')

    print("3 SVM RBF - Breast Cancer")
    breastCancerSVM = SVM(breastCancerDataSet, False)
    breastCancerSVM.validate()
    print ('--------------------------------------------------------')


    print("3 SVM Linear AUC - Breast Cancer")
    breastCancerSVMAUC = SVM_AUC(breastCancerDataSet)
    breastCancerSVMAUC.validate()
    print ('--------------------------------------------------------')

    print("3 SVM RBF AUC - Breast Cancer")
    breastCancerSVMAUC = SVM_AUC(breastCancerDataSet, False)
    breastCancerSVMAUC.validate()
    print ('--------------------------------------------------------')

    
    
    
    print("3 SVM Linear - Spambase")
    spambaseSVM = SVM(spambaseDataSet)
    spambaseSVM.validate()
    print ('--------------------------------------------------------')

    print("3 SVM RBF - Spambase")
    spambaseSVM = SVM(spambaseDataSet, False)
    spambaseSVM.validate()
    print ('--------------------------------------------------------')


    print("3 SVM Linear AUC - Spambase")
    spambaseSVMAUC = SVM_AUC(spambaseDataSet)
    spambaseSVMAUC.validate()
    print ('--------------------------------------------------------')

    print("3 SVM RBF AUC - Spambase")
    spambaseSVMAUC = SVM_AUC(spambaseDataSet, False)
    spambaseSVMAUC.validate()
    print ('--------------------------------------------------------')


#4
    print("4 Multiclass Linear - Wine")
    wineMulticlassSVM = SVMMulticlass(wineDataSet)
    wineMulticlassSVM.validate()
    print ('--------------------------------------------------------')

    print("4 Multiclass RBF - Wine")
    wineMulticlassSVM = SVMMulticlass(wineDataSet, False)
    wineMulticlassSVM.validate()
    print ('--------------------------------------------------------')

    print("4 Multiclass Linear AUC - Wine")
    wineMulticlassSVMAUC = SVMMulticlassAUC(wineDataSet)
    wineMulticlassSVMAUC.validate()
    print ('--------------------------------------------------------')

    print("4 Multiclass RBF AUC - Wine")
    wineMulticlassSVMAUC = SVMMulticlassAUC(wineDataSet, False)
    wineMulticlassSVMAUC.validate()
    print ('--------------------------------------------------------')



