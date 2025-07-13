import numpy as np


rng = np.random.default_rng(420)


#data = np.loadtxt(open('data/training_spam.csv'), delimiter=',').astype(int)  #data is (1000, 55)
#print(data.shape)

#training_spam = np.loadtxt(open("data/testing_spam.csv"), delimiter=",").astype(int)

labels = training_spam[:, 0] 

features = training_spam[:, 1:] 


#print(features.shape, labels.shape, x.shape)

class SpamClassifier:
    def params(self):

        self.W1 = rng.normal(0, np.sqrt(2. / 54), size=(54, 32))  # 32 neurons in the hidden layer
        self.B1 = np.zeros((1,32)) 

        self.W2 = rng.normal(0, np.sqrt(2. / (32)), size=(32, 1))  
        self.B2 = np.zeros((1,1))
   
    def ReLU(self, Z):
        return np.maximum(Z, 0)
   
    def ReLU_deriv(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
   
    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)


    def forward_propagation(self, features):
        self.Z1 = np.dot(features, self.W1) + self.B1  # (1000,54) * (54,32) = (1000,32)
        self.A1 = self.ReLU(self.Z1)         # 
        self.Z2 = np.dot(self.A1, self.W2) + self.B2   # (1000,1)
        self.A2 = self.sigmoid(self.Z2) 
   
    def loss_function(self, labels):
        m = labels.shape[0]
        labels = labels.reshape(-1, 1) 
        epsilon = 1e-15  # To avoid log(0)
        A2Clipped = np.clip(self.A2, epsilon, 1 - epsilon)
        loss = -np.mean(labels * np.log(A2Clipped) + (1 - labels) * np.log(1 - A2Clipped))
        return loss


    def backward_prop(self, features, labels, learning_rate):
        m = labels.shape[0]
        labels = labels.reshape(-1, 1) 

        
        self.dZ2 = self.A2 - labels  # (1000,1)

        self.dW2 = np.dot(self.A1.T , self.dZ2) / m    # A1 is (1000,32) , W2 has to be (1,32) so dW2 is (1,32)
        self.db2 = np.sum(self.dZ2, axis=0, keepdims=True) / m 

        self.dA1 = np.dot(self.dZ2, self.W2.T)  # ahs to be 1000,32 
        self.dZ1 = self.dA1 * self.ReLU_deriv(self.Z1)


        self.dW1 = np.dot(features.T,self.dZ1) / m #32,54
        self.db1 = np.sum(self.dZ1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * self.dW2   
        self.B2 -= learning_rate * self.db2
        self.W1 -= learning_rate * self.dW1
        self.B1 -= learning_rate * self.db1


    def train(self, features, labels, learning_rate, epochs):

        initial_learning = learning_rate

        for epoch in range(epochs):

            learning = initial_learning * np.exp(-0.0003 * epoch) 


            self.forward_propagation(features)


            loss = self.loss_function(labels)


            self.backward_prop(features, labels, learning)

            #prediction = (self.A2 >= 0.5).astype(int)
            #train_acc = np.mean(prediction.flatten() == labels)
            #print(f"Epoch {epoch + 1}/{epochs}, Learning_rate {learning} , train acc {train_acc} , Loss: {loss:.6f}")


    def predict(self, features):
        self.forward_propagation(features) 
        predictions = (self.A2 >= 0.5).astype(int) 
        return predictions.flatten() 
    
    def save_weights(self, filename):
        np.savez(filename,W1=self.W1, B1=self.B1,W2=self.W2, B2=self.B2)

    def load_weights(self, filename):
        data = np.load(filename)
        self.W1 = data['W1']
        self.B1 = data['B1']
        self.W2 = data['W2']
        self.B2 = data['B2']

    # to run this code properly. my npz file called "weights_and_biases.npz" must be in the same folder as this ipynb file

    # instructions on how to compute the original biases and weights (how to do training) are in the comment below

    # if you want to compute the original weights then uncomment classifier.train(features, labels, learning_rate=0.02, epochs=3500)
    # inside the create_flassifier function

    # uncomment classifier.save_weights("weights_and_biases.npz") and then comment classifier.load_weights("weights_and_biases.npz")


def create_classifier():
    classifier = SpamClassifier()
    classifier.params()
    #classifier.train(features, labels, learning_rate=0.02, epochs=3500)
    #classifier.save_weights("weights_and_biases.npz")
    classifier.load_weights("weights_and_biases.npz")
    return classifier

classifier = create_classifier()
