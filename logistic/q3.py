import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import sys

args = len(sys.argv)
train_path = sys.argv[1]
test_path = sys.argv[2]

X = pd.read_csv(train_path+'/X.csv',names=['X1','X2'])
Y = pd.read_csv(train_path+'/Y.csv',names=['Y'])

rows,col = X.shape

feature_1 = X.iloc[:,0]
mean1 = np.mean(feature_1,axis=0)
std_dev1 = np.std(feature_1)

feature_2 = X.iloc[:,1]
mean2 = np.mean(feature_1,axis=0)
std_dev2 = np.std(feature_2)



feature_1 = np.array([(feature_1-mean1)/std_dev1]).transpose()
feature_2 = np.array([(feature_2-mean2)/std_dev2]).transpose()
ones = np.ones((rows,1))
data = np.append(ones,feature_1,axis=1)
data = np.append(data,feature_2,axis=1)

X = data


# calculating Hessian.
def h(x,theta):
    vector = np.dot(x,theta)
    value = 1/(1+np.exp(-vector))
    return value




def Hessian(x,theta):
    arr = h(x,theta)*(1-h(x,theta))
    vector = np.array(arr[:,0])
    D = np.diag(vector)
    ##D is 99*99
    D = np.dot(x.transpose(),D)
    #x.transpose() = 3*99, * D (99*99) = Result(3*99)
    hessian = np.dot(D,x)
    #D(3*99) x(99*3)= hessian(3*3)
    return hessian 
                      
                      
                      

# Finding Gradient of L
def grad_L(x,y,theta):
    h_theta = h(x,theta)
    diff = (y-h_theta)
    #x.T is 3*100, diff = 100*1, gradient = 3*1
    gradient = np.dot(x.transpose(),diff)
    return gradient

#Implementing Newton's Method
Theta = np.array([[0,0,0]]).transpose()
hessian_inv = np.linalg.inv(Hessian(X,Theta))
gradient_L = grad_L(X,Y,Theta)
Theta_new = Theta + np.dot(hessian_inv,gradient_L)
print(Theta_new)

# Plot the training data (your axes should be x1 and x2 , corresponding to the two coordinates of the inputs, and you should use a different symbol for each point plotted to indicate whether that example had label 1 or 0). Also plot on the same figure the decision boundary fit by logistic regression. (i.e., this should be a straight line showing the boundary separating the region where h(x) > 0.5 from where h(x) ≤ 0.5.)

prediction = h(X,Theta_new)
theta_0 = Theta_new[0][0]
theta_1 = Theta_new[1][0]
theta_2 = Theta_new[2][0]

y_plot = -((theta_0+theta_1*feature_1)/theta_2)

colors = {0:'red', 1:'green'}
markers = {0:'*',1:'^'}


def change(vector):
    for i in range(0,vector.size):
        if(vector[i]<0.5):
            vector[i]=0
        else:
            vector[i]=1
    return vector
vec = change(prediction)
df = pd.DataFrame(vec,columns=['prediction'])

plt.scatter(feature_1, feature_2, c=df['prediction'].map(colors))
plt.plot(feature_1,y_plot)
plt.xlabel( "X_Values" , size = 12 )
plt.ylabel( "Y_Values)" , size = 12 )
plt.title( "Classification" , size = 24 )
#plt.show()

def prediction_fun(test_path):
    x_t = pd.read_csv(test_path+'/X.csv',names=['X','Y'])
    r,c = x_t.shape
    ones1 = np.ones((r,1))
    t_data = np.append(ones1,x_t,axis=1)
    res = h(t_data,Theta_new)
    for i in range(0, res.size):
        if(res[i]<0.5):
            res[i]=0
        else:
            res[i]=1
    save_path ='./'
    vari_p = os.path.join(save_path,'result_3.txt')

    file = open(vari_p,'w+')
    np.savetxt(vari_p,res)
    file.close()

    #predicted_val= np.dot(test,theta_final_1)


prediction_fun(test_path)    

# import seaborn as sns
# data_f=pd.DataFrame(np.column_stack((feature_1,feature_2,prediction)),columns=['X_Values','Y_Values','Class'])
# data_f2 = data_f
# data_f2['y']= y_plot

# #df1 = sns.load_dataset("data_f")
# sns.scatterplot(data=data_f,x="X_Values",y="Y_Values",hue="Class",style="Class")
# sns.lineplot(data=data_f2, x="X_Values", y="y",color='green')

