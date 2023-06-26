import torch
import numpy as np
import torch.nn as nn

class PhysicsInformedBarModel:
    """A class used for the definition of Physics Informed Models for one dimensional bars."""

    def __init__(self, x, E, A, L, u0, dist_load):
        """Construct a PhysicsInformedBar model
        Enter your code
        Task : initialize required variables for the class
        """
        self.x = x
        self.E = E
        self.A = A
        self.L = L
        self.u0 = u0
        self.dist_load = dist_load
        self.model = nn.Sequential(nn.Linear(1, 40),nn.Tanh(),nn.Linear(40, 40),nn.Tanh(),nn.Linear(40, 1))
        self.mseb = None
        self.mseu = None
        self.total_loss = None
        self.optimizer = None

    def costFunction(self, x, u_pred):
        """Compute the cost function.
        This function takes input x and model predicted output u_pred to compute loss

        Params:
            x - spatial value
            u_pred - NN predicted displacement value
            differential_equation_loss - calculated PDE residual loss
            boundary_condition_loss - calculated boundary loss
        """

        u_true = torch.tensor([self.u0, self.u0], requires_grad=True, dtype=torch.float32)
        
        boundary_condition_loss = torch.mean( (u_pred-u_true)**2 )

        u = self.model(self.x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True,  retain_graph=True)[0]
        sum_ = self.E*self.A*u_xx + self.dist_load(x)
        differential_equation_loss = torch.mean(sum_**2)

        return differential_equation_loss, boundary_condition_loss


    def get_displacements(self, x):
        """Get displacements.
        This function is used while inference (you can even use in your training phase if needed.
        It takes x as input and returns model predicted displacement)
        Params:
            x - input spatial value
            u - model predicted displacement
        """

        "Enter your code"
        u = self.model(x)   # predict
        return u

    def train(self, epochs, optimizer, **kwargs):
        """Train the model.
        This function is used for training the network. While updating the model params use "costFunction"
        function for calculating loss
        Params:
            epochs - number of epochs
            optimizer - name of the optimizer
            **kwarhs - additional params

        This function doesn't have any return values. Just print the losses during training

        """

         # Select optimizer
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), **kwargs)
        
        # Initialize history arrays. Save the values to plot wrt epochs
        self.mseb = np.zeros(epochs)
        self.mseu = np.zeros(epochs)
        self.total_loss = np.zeros(epochs)
        
        # Training loop
        for epoch in range(epochs):
            xb = torch.tensor([0., 1.], requires_grad=True, dtype=torch.float32).unsqueeze(1)
            u_pred = self.get_displacements(xb)
            differential_equation_loss, boundary_condition_loss = self.costFunction(self.x,u_pred)
            total_loss = boundary_condition_loss + differential_equation_loss
            self.mseb[epoch] = boundary_condition_loss
            self.mseu[epoch] = differential_equation_loss
            self.total_loss[epoch] = total_loss

            # Set gradients to zero.
            self.optimizer.zero_grad()
            
            # Compute gradient (backwardpropagation). Basically find the derivative of cost function wrt to weights and biases.
            total_loss.backward(retain_graph=True)
            
            # Update parameters. In previous step it calculate the weights and biases but did not update.
            def closure():
                # Set gradients to zero.
                #torch.autograd.set_detect_anomaly(True)
                self.optimizer.zero_grad()
                l1,l2 = self.costFunction(self.x,self.get_displacements(xb))
                total_loss =l1+l2
                # Compute gradient (backwardpropagation). Basically find the derivative of cost function wrt to weights and biases.
                total_loss.backward(retain_graph=True)
                return total_loss
            
            self.optimizer.step(closure=closure)         
            
            if epoch < epochs:
                #print("Cost function: " + cost.detach().numpy())
                print(f'Epoch: {epoch+1}, Mseu: {differential_equation_loss.detach().numpy()},  Mseb: {boundary_condition_loss.detach().numpy()}, Cost: {total_loss.detach().numpy()}')
            
            #plt.scatter(epoch,cost.detach().numpy()) # real time plot
            
        #plt.show()
        return epoch, total_loss.detach().numpy()
    
   