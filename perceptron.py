import numpy as np
import itertools
class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights = np.random.randn(self.number_of_classes,self.input_dimensions+1)
        #raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))
        #weight_matrix[row,col]=>weight_matrix[no.ofclasses,inputdimension+1] , +1 for the bias
        #raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        Xnew=np.insert(X,0,1,axis=0)#inserting row of one
        #inserting x0 = 1 as an input(x0,x1...xn)
        net=np.dot(self.weights,Xnew) #a=Wp+b
        op=np.where(net>=0.0,1,0) #implementing hardlimit function(transfer function).
        return op
        
        
        #raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)
        #raise Warning("You must implement print_weights")

    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        
        Xnew=np.insert(X,0,1,axis=0) #size -> (3,4)
        for _ in range(num_epochs):
            for i in range(Xnew.shape[1]):
                a=np.reshape(self.predict(X[0:,i]),self.number_of_classes,1)
                t=np.reshape(Y[:,i],self.number_of_classes,1)
                e=np.array(t-a)
                p= np.array(Xnew[:, i])
                self.weights=self.weights+alpha*e.reshape(self.number_of_classes,1)*p
                #w_new=w_old+learningrate*(t-a)*p'
                #used reshape just to make sure that the shape of array does not change with every operation
                
        #raise Warning("You must implement train")

        

                
                


    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        
        error_count=0
        error=np.array(self.predict(X))
        err=np.transpose(Y-error)
        #print(err)
        #print('hello')
        #print(error)
        
        for i in err:
            if np.any(i)!=0: # if any (target - actual) value does not have value 0,increment the error counter
                error_count+=1
        return error_count/X.shape[1] #returning the calculated error
        
        

        
        #raise Warning("You must implement calculate_percent_error")

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """

    input_dimensions = 2
    number_of_classes = 2

    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)#change the num_epoch just for experiment
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)