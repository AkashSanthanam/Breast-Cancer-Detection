import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


# ----------------------------- DATA PROCESSING ---------------------------------------------------

data = pd.read_csv('data1.csv')

# drop first and last column
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
#segregate inputs and targets

#inputs
X = data.iloc[:, 1:]

#add column for bias
X.insert(loc=len(X.columns),column="bias", value=1)
X_features = X.to_numpy()


#converting categorical variables to integers
# - this is same as using one hot encoding from sklearn
#benign = -1, melignant = 1
category_dict = {'B': -1.0,'M': 1.0}
#transpose to column vector
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)

 # ------------------------------------------------------------------------------------------------


class svm_():
    def __init__(self,learning_rate,epoch,C_value,X,Y):

        #initialize the variables
        self.input = X
        self.target = Y

        # To plot training and validation for SGD and mini-batch SGD
        self.training_losses = []
        self.validation_losses = []

        # To plot sampling training and valdiation
        self.sampling_training = []
        self.sampling_validation = []


        self.learning_rate =learning_rate
        self.epoch = epoch
        self.samples = 0
        self.C = C_value

        # To scale the training and validation
        self.scalar = StandardScaler().fit(self.input)

        # Was used to determine stopping pooint
        self.early_stopping_point = [0,0,0]

        #initialize the weight matrix based on number of features
        # bias and weights are merged together as one matrix
        # you should try random initialization

        self.weights = np.random.randn(X.shape[1])

    def pre_process(self,):

        #using StandardScaler to normalize the input
        X_ = self.scalar.transform(self.input)

        Y_ = self.target

        return X_,Y_

    # the function return gradient for 1 instance -
    # stochastic gradient decent
    def compute_gradient(self,X,Y):
        # organize the array as vector
        X_ = np.array([X])

        # hinge loss
        hinge_distance = 1 - (Y * np.dot(X_,self.weights))

        total_distance = np.zeros(len(self.weights))
        # hinge loss is not defined at 0
        # is distance equal to 0
        if max(0, hinge_distance[0]) == 0:
            total_distance += self.weights
        else:
            total_distance += self.weights - (self.C * Y[0] * X_[0])

        return total_distance

    def compute_loss(self,X,Y):
        # calculate hinge loss
        # hinge loss implementation- start
        X_ = np.array([X])
        margin = 1 - (Y * np.dot(X_, self.weights))

        maxmargins = np.maximum(0, margin)
        hinge_loss = self.C * np.mean(maxmargins)
        regularization = (self.C * 0.5) * (np.linalg.norm(self.weights) ** 2)
        total_loss = hinge_loss + regularization
        return total_loss

        # hinge loss implementatin - end
    def stochastic_gradient_descent(self,X_train, Y_train, X_val, Y_val):

        samples=0
        prev_val_loss = float('inf')

        # So for SGD, I computed the absolute difference for every 2 epochs
        two_epochs_val_loss = float('inf')
        diff = prev_val_loss

        # To track the best model once early stopping triggers
        best_weights = None

        # General Threshold 
        threshold = 1e-4
        min_training = float('inf')

        print("Training Started")

        # execute the stochastic gradient descent function for defined epochs
        for epoch in range(self.epoch):

            # shuffle to prevent repeating update cycles
            features, output = shuffle(X_train, Y_train)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            training_loss = self.compute_loss(features, output)
            val_loss = self.compute_loss(X_val, Y_val)
            diff = val_loss - two_epochs_val_loss
            #print epoch if it is equal to thousand - to minimize number of prints
            #check for convergence -start
            if epoch % (self.epoch // 10) == 0:
               print("Epoch: ", epoch, "Diff: ", diff)
               self.training_losses.append(training_loss)
               self.validation_losses.append(val_loss)


            # Checking if the difference is within the threshold. What that means is the validation has plateau and isn't making any significant changes so there is no need to continue training model which prevents overfitting. I notice without tht break statement that convergence would usually sit around 260-400, so I wanted to get past that epoch interval to prevent the algorithm from stopping too early

            if abs(diff) < threshold and epoch > 300:
              if min_training > training_loss:
                 min_training = training_loss
                 print("min number of epochs: ", epoch, "Training: ", training_loss, "Validation: ", val_loss)

                 # Storing the point
                 self.early_stopping_point[0] = training_loss
                 self.early_stopping_point[1] = val_loss
                 self.early_stopping_point[2] = epoch


                # Storing the weights
                 best_weights = self.weights.copy()

                # Feel free to comment this out if you wish to see the graph beyond the stopping iterations
                 break

            # Storing the losses during previous epochs
            two_epochs_val_loss = prev_val_loss
            prev_val_loss = val_loss


            #check for convergence - end

            # below code will be required for 

            samples+=1

        self.weights = best_weights
        self.samples = samples

        print("Training ended...")
        # print("Final weights are: {}".format(self.weights))

        # below code will be required for 
        print("The minimum number of samples used are:",samples)

    def mini_batch_gradient_descent(self,X_train,Y_train,X_val, Y_val, batch_size):

        # mini batch gradient decent implementation - start

        num_samples = X.shape[0]
        samples = 0  

        print("Training Started")
        for epoch in range(self.epoch):

            # Shuffle the data at the beginning of each epoch
            features, output = shuffle(X_train, Y_train)

            # Iterate over the dataset in batches
            for i in range(0, num_samples, batch_size):
                # Extract the current mini-batch
                X_batch = features[i:i + batch_size]
                Y_batch = output[i:i + batch_size]

                # Initialize batch gradient accumulator
                batch_gradient = np.zeros(self.weights.shape)

                # Compute the gradient for each sample in the mini-batch
                for j in range(len(X_batch)):
                    gradient = self.compute_gradient(X_batch[j], Y_batch[j])
                    batch_gradient += gradient  # Accumulate the gradients

                # Compute the average gradient for the batch
                batch_gradient /= batch_size

                # Update the weights based on the averaged gradient
                self.weights -= self.learning_rate * batch_gradient

                # Increment the sample counter
                samples += batch_size

            # Calculate and print the loss every 1000 epochs (or another interval)
            if epoch % (self.epoch // 10) == 0:
                print("Epoch", epoch)
                loss = self.compute_loss(features, output)
                val_loss = self.compute_loss(X_val, Y_val)
                self.training_losses.append(loss)
                self.validation_losses.append(val_loss)

        # You may implement convergence check here if needed
        
        # mini batch gradient decent implementation - end

        print("Training ended...")
        print("weights are: {}".format(self.weights))


    def sampling_strategy(self, X_train,Y_train, X_val, Y_val):
        
        #implementation of sampling strategy - start
        initial_indices = np.random.choice(len(X_train), size=50, replace=False)
        X_train_initial = X_train[initial_indices]
        Y_train_initial = Y_train[initial_indices]

        # Performing predictions on the remaining samples   
        remaining_indices = list(set(range(len(X_train))) - set(initial_indices))
        X_remaining = X_train[remaining_indices]
        Y_remaining = Y_train[remaining_indices]  

        for epoch in range(self.epoch):
            print(f"Iteration {epoch + 1}")

            if not remaining_indices:
                break  # Exit the loop if no more samples are remaining

            # Calculating the loss for all remaining samples and select the one with the smallest loss
            losses = []
            for i in range(X_remaining.shape[0]):
                loss = self.compute_loss(X_remaining[i], Y_remaining[i])  # Use the existing loss function
                losses.append(loss)

            # Finding the index of the sample with the smallest loss
   
            min_loss_index = np.argmin(losses)

            next_sample = X_remaining[min_loss_index]
            next_label = Y_remaining[min_loss_index]

            # Train the classifier further with the selected sample
            self.stochastic_gradient_descent(np.array([next_sample]), np.array([next_label]), np.array([next_sample]), np.array([next_label]))

            # Remove the selected sample from the remaining set
            selected_index = remaining_indices[min_loss_index] 
            remaining_indices.remove(selected_index)

            X_remaining = X_train[remaining_indices]
            Y_remaining = Y_train[remaining_indices]

            ### Once I trained the model with the best training sample (the one with lowest loss) I compute the training loss and validation loss with the current weights for the training and validation set

            if epoch % (self.epoch // 10) == 0:
               print("Epoch: ", epoch)
               self.sampling_training.append(self.compute_loss(X_train, Y_train))
               self.sampling_validation.append(self.compute_loss(X_val, Y_val))


            # Here I added a performance threshold based on the remaining subset. The reason why I commented this out is because the model performs the best in the intial iterations as that is when best samples are being used from the random sample subset , so it will break. However this prevents it from showing the full graph for the loss. You can uncomment out this code and see how the accuracy compares on the held-out test set in the section. 
            
            # accuracy, precision, recall = self.predict(X_remaining,Y_remaining)
            # if accuracy > 0.90 and precision > 0.90 and recall > 0.90 and epoch > 50:
            #   print(f"Performance threshold met: accuracy={accuracy}, precision={precision}, recall={recall}. Stopping training.")
            #   break
           
      
        #implementation of sampling strategy - start
        return X_train_initial, Y_train_initial

    def predict(self,X_test,Y_test):

        #compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]

        #compute accuracy
        accuracy= accuracy_score(Y_test, predicted_values)
        print("Accuracy on test dataset: {}".format(accuracy))

        #compute precision - start
        precision = precision_score(Y_test, predicted_values)
        print("Precision on test dataset: {}".format(precision))
        #compute precision - end

        #compute recall - start
        recall = recall_score(Y_test, predicted_values)
        print("Recall on test dataset: {}".format(recall))
        #compute recall - end
        return accuracy, precision, recall



def experiment_1(X_train,y_train, X_val, y_val):
    C = 1
    learning_rate = 0.0035
    epoch = 2500

    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    scalar = StandardScaler().fit(X_train)
    X_norm = scalar.transform(X_train)
    X_Val_Norm = scalar.transform(X_val)

    # # train model
    my_svm.stochastic_gradient_descent(X_norm, y_train, X_Val_Norm, y_val)
    print("SGD Scores")
    my_svm.predict(X_Val_Norm,y_val)


    return my_svm

 # ---------------------------- COMMENT THIS SECTION OUT TO RUN ------------------------------
# First split: 75% training, 25% (validation)
# X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.25, random_state=42)

# my_svm = experiment_1(X_train, y_train, X_test, y_test)


# plt.figure(figsize=(10, 6))


# plt.plot(range(len(my_svm.training_losses)), my_svm.training_losses, label='Training Loss', color='blue')
# plt.plot(range(len(my_svm.validation_losses)), my_svm.validation_losses, label='Validation Loss', color='red')


# plt.xlabel('Epoch')
# plt.ylabel('Loss')


# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()

 # ------------------------------------------------------------------------------------------------


def experiment_2(X_train,y_train, X_Val, y_val):
    C = 1
    learning_rate = 0.0085
    epoch = 1000

    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    scalar = StandardScaler().fit(X_train)
    X_norm = scalar.transform(X_train)
    X_Val_Norm = scalar.transform(X_Val)

    # # train model
    my_svm.mini_batch_gradient_descent(X_norm, y_train, X_Val_Norm, y_val, 5)

    print("Mini-Batch Scores")

    my_svm.predict(X_Val_Norm,y_val)



    return my_svm

 # ---------------------------- COMMENT THIS SECTION OUT TO RUN ------------------------------
# First split: 75% training, 25% (validation)
# X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.25, random_state=42)

# # Setting up models for Mini-Batch and SGD
# my_svm = experiment_2(X_train, y_train, X_test, y_test)
# my_svm2 = experiment_1(X_train, y_train, X_test, y_test)

# plt.figure(figsize=(10, 6))

# # Plotting Mini-Batch

# plt.plot(range(len(my_svm.training_losses)), my_svm.training_losses, label='Training Loss (Mini-Batch)', color='blue')
# plt.plot(range(len(my_svm.validation_losses)), my_svm.validation_losses, label='Validation Loss (Mini-Batch)', color='red')

# # Plotting SGD
# plt.plot(range(len(my_svm2.training_losses)), my_svm2.training_losses, label='Training Loss (SGD)', color='green')
# plt.plot(range(len(my_svm2.validation_losses)), my_svm2.validation_losses, label='Validation Loss (SGD)', color='yellow')

# plt.xlabel('Epoch')
# plt.ylabel('Loss')


# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()


 # ------------------------------------------------------------------------------------------------

def experiment_3(X_train,y_train, X_Val, y_val):
    C = 1
    learning_rate = 0.001
    epoch = 2500

    #intantiate the support vector machine class above
    my_svm = svm_(learning_rate=learning_rate,epoch=epoch,C_value=C,X=X_train,Y=y_train)

    #pre preocess data
    scalar = StandardScaler().fit(X_train)
    X_norm = scalar.transform(X_train)
    X_Val_Norm = scalar.transform(X_Val)

    #train model
    my_svm.sampling_strategy(X_norm, y_train, X_Val_Norm, y_val)

    print("Sampling Scores")
    my_svm.predict(X_Val_Norm,y_val)

    return my_svm


 # ---------------------------- COMMENT THIS SECTION OUT TO RUN------------------------------
# First split: 75% training, 25% (validation)
# X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.25, random_state=42)
# my_svm = experiment_3(X_train, y_train, X_test, y_test)


# plt.figure(figsize=(10, 6))


# plt.plot(range(len(my_svm.sampling_training)), my_svm.sampling_training, label='Training Loss (Sampling)', color='blue')
# plt.plot(range(len(my_svm.sampling_validation)), my_svm.sampling_validation, label='Validation Loss (Sampling)', color='red')


# plt.xlabel('Epoch')
# plt.ylabel('Loss')


# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()


 # ------------------------------------------------------------------------------------------------
