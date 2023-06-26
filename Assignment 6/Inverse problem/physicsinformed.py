import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
class InversePhysicsInformedBarModel:
    """
    A class used for the definition of the data driven approach for Physics Informed Models for one dimensional bars. 
    EA is estimated.
    """

    def __init__(self, x, u, L, dist_load):
        """Construct a InversePhysicsInformedBarModel model"""

        self.x = x
        self.u=u
        self.L = L
        self.dist_load = dist_load
        self.model = nn.Sequential(nn.Linear(2, 20),nn.Tanh(),nn.Linear(20, 1))
        self.loss = None
        self.optimizer = None

    def predict(self, x, u):
        """Predict parameter EA of the differential equation."""

        '''
        Params: 
            x - input spatial value
            u - input displacement value at x
            ea - model predicted value
        '''

        '''
        Enter your code
        '''
        X = torch.cat([x,u],axis=1)
        ea = self.model(X)
        return ea

    def cost_function(self, x, u):
        """Compute the cost function."""

        '''
        Params:
            x - input spatial value
            u - displacement value at x
            EA_pred - model predicted EA value
            differential_equation_loss - calculated physics loss
        '''

        '''
        Enter your code
        '''
        EA_pred = self.predict(x,u)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True,  retain_graph=True)[0]

        EA_x = torch.autograd.grad(EA_pred, x, grad_outputs=torch.ones_like(EA_pred), create_graph=True, retain_graph=True)[0]

        differential_equation_loss = torch.mean((EA_pred*u_xx + EA_x*u_x + self.dist_load(x))**2)

        return differential_equation_loss
    
    def train(self, epochs, optimizer, **kwargs):
        """Train the model."""

        '''
        This function is used for training the network. While updating the model params use "cost_function" 
        function for calculating loss
        Params:
            epochs - number of epochs
            optimizer - name of the optimizer
            **kwarhs - additional params

        This function doesn't have any return values. Just print the losses during training
        
        '''

         # Select optimizer
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)
        
        # Initialize history arrays. Save the values to plot wrt epochs
        self.loss = np.zeros(epochs)
        
        # Training loop
        for epoch in range(epochs):
            total_loss= self.cost_function(self.x,self.u)
            self.loss[epoch] = total_loss

            # Set gradients to zero.
            self.optimizer.zero_grad()
            
            # Compute gradient (backwardpropagation). Basically find the derivative of cost function wrt to weights and biases.
            total_loss.backward(retain_graph=True)
            
            # Update parameters. In previous step it calculate the weights and biases but did not update.
            def closure():
                # Set gradients to zero.
                self.optimizer.zero_grad()
                total_loss = self.cost_function(self.x,self.u)
                # Compute gradient (backwardpropagation). Basically find the derivative of cost function wrt to weights and biases.
                total_loss.backward(retain_graph=True)
                return total_loss
            
            self.optimizer.step(closure=closure)          
            
            if epoch < epochs:
                #print("Cost function: " + cost.detach().numpy())
                print(f'Epoch: {epoch+1}, Cost: {total_loss.detach().numpy()}, Differential Equation Loss: {total_loss.detach().numpy()}')
            
            #plt.scatter(epoch,cost.detach().numpy()) # real time plot
            
        #plt.show()
        return epoch, total_loss.detach().numpy()
