# # Solution of 2D Forward Problem of Linear Elasticity for Plane Stress Boundary Value Problem using Physics-Informed Neural Networks (PINN)

# ### Solve the 2D plane stress problem of elasticity using PINN using PDE and BC losses. Use stochastic gradient descent for training. Penalize the boundary condition loss with a factor of 10,000.


import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

torch.manual_seed(123456)
np.random.seed(123456)



E = 1                                       # Young's Modulus
nu = 0.3                                    # Poisson Ratio
G =  E/(2*(1+nu))                                       # Shear modulus



Total_loss = []
PDE_loss = []
BC_loss = []


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define your model here (refer: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
        self.Nnet = nn.Sequential()
        self.Nnet.add_module('Hidden_layer_1', nn.Linear(2,30))    # First linear layer
        self.Nnet.add_module('Tanh_layer_1', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_2', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_2', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_3', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_3', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_4', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_4', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_5', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_5', nn.Tanh())
        self.Nnet.add_module('Output Layer', nn.Linear(30,2))
        
        print(self.Nnet)                                        # Print model summary

    # Forward Feed
    def forward(self, x):
        y = self.Nnet(x)
        return y

    # PDE and BCs loss
    def loss(self, x, x_b, b_u , b_v , epoch):
        y = self.forward(x)   # Interior Solution
        y_b = self.forward(x_b)   # Boundary Solution
        u_b, v_b = y_b[:, 0], y_b[:, 1]     # u and v boundary
        u, v = y[:, 0], y[:, 1]   # u and v interior

        # Calculate Gradients
        # Gradients of deformation in x-direction
        u_g =  gradients(u, x)[0]                    # Gradient of u, Du = [u_x, u_y]
        u_x, u_y = u_g[:, 0], u_g[:, 1]             # [u_x, u_y]
        u_xx = gradients(u_x , x)[0][:, 0]           # Second derivative, u_xx
        u_xy = gradients(u_x , x)[0][:, 1]           # Mixed partial derivative, u_xy
        u_yy = gradients(u_y , x)[0][:, 1]           # Second derivative, u_yy

        # Gradients of deformation in y-direction
        v_g = gradients(v , x)[0]                    # Gradient of v, Du = [v_x, v_y]
        v_x, v_y = v_g[:, 0], v_g[:, 1]             # [v_x, v_y]
        v_xx = gradients(v_x , x)[0][:, 0]           # Second derivative, v_xx
        v_yx = gradients(v_y , x)[0][:, 0]
        v_yy = gradients(v_y , x)[0][:, 1]

        f_1 = torch.sin(2*np.pi*x[:,1])*torch.sin(2*np.pi*x[:,0]) # Define body force for PDE-1
        f_2 =  torch.sin(2 * np.pi * x[:, 1]) + torch.sin(np.pi * x[:, 0])   # Define body force for PDE-2
        
        loss_1 = (G*(u_xx + u_yy) + G*((1+nu)/(1-nu))*(u_xx + v_yx) + f_1)   # Define loss for PDE-1
        loss_2 = (G*(v_xx + v_yy) + G*((1 + nu) / (1 - nu))*(u_xy + v_yy) + f_2)   # Define loss for PDE-2

        loss_PDE = torch.mean((loss_1) ** 2) + torch.mean((loss_2) ** 2)
        loss_bc = torch.mean((u_b - b_u) ** 2) + torch.mean((v_b - b_v) ** 2)

        TotalLoss = loss_PDE + (loss_bc)*10000
        
        Total_loss.append(TotalLoss.detach().numpy())
        PDE_loss.append(loss_PDE.detach().numpy())
        BC_loss.append(loss_bc.detach().numpy()*10000)
        
        if epoch % 100 == 0:
            print(f'epoch {epoch}: loss_pde {loss_PDE:.8f}, loss_bc {loss_bc:.8f}')
            
        return TotalLoss



def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),allow_unused=True, create_graph=True)


# Define model parameters here
device = ("cuda" if torch.cuda.is_available() else "cpu")
epochs = 2000                                            # Number of epochs
lr = 0.0005                                                  # Learning Rate



# Load the collocation point data
data = scipy.io.loadmat('interior_points.mat')                     # Import interior points data
x = data['x']                     # Partitioned x coordinates
y = data['y']                            # Partitioned y coordinates
interior_points = np.concatenate((x, y), axis=1)                       # Concatenate (x,y) iterior points

boundary = scipy.io.loadmat('boundary_points.mat')                     # Import boundary points
x_boundary = boundary['x_bdry']                   # Partitioned x boundary coordinates
y_boundary = boundary['y_bdry']                # Partitioned y boundary coordinates

boundary_points = np.concatenate((x_boundary, y_boundary), axis=1)       # Concatenate (x,y) boundary points

u_bound_ext = np.zeros((len(boundary_points)))          # Dirichlet boundary conditions



## Define data as PyTorch Tensor and send to device
xy_f_train = torch.tensor(interior_points, requires_grad=True, dtype=torch.float32).to(device)
xy_b_train = torch.tensor(boundary_points, requires_grad=True, dtype=torch.float32).to(device)

# Define the boundary condition values
u_b_train = torch.tensor(u_bound_ext, dtype=torch.float32).to(device)
v_b_train = torch.tensor(u_bound_ext, dtype=torch.float32).to(device)

# Initialize model
model = Model().to(device)

# Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)



# Training
def train(epoch):
    model.train()

    def closure():
        optimizer.zero_grad()
        loss = model.loss(xy_f_train,xy_b_train,u_b_train,u_b_train,epoch)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    if epoch % 100 == 0:
            print(f'epoch {epoch}: Total loss {loss:.8f}')
    return loss



for epoch in range(epochs):
    train(epoch+1)


# ## 1. Plot the displacement field contours in 𝑥− and 𝑦− directions after training the model.


ui_preds = model.forward(xy_f_train)
ub_preds = model.forward(xy_b_train)
u_preds = torch.cat((ui_preds, ub_preds), dim = 0).detach().numpy()
u_plot = u_preds[:,0]
v_plot = u_preds[:,1]
xy_all = torch.cat((xy_f_train, xy_b_train), dim=0).detach().numpy()
x = xy_all[:,0]
y = xy_all[:,1]    
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Interpolate the data onto the grid
U = griddata((x, y), u_plot, (X, Y), method='linear')
V = griddata((x, y), v_plot, (X, Y), method='linear')

# Plot the results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
im1 = ax1.imshow(U, cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
im2 = ax2.imshow(V, cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

ax1.set_title('Displacement in x direction')
ax2.set_title('Displacement in y direction')
plt.show()


# ##  2. Plot the total loss, PDE loss and BC loss versus epochs.


# Plot total net loss vs epochs
plt.plot(Total_loss)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cost History')



# Plot PDE loss vs epochs
plt.plot(PDE_loss)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('PDE Loss')
plt.title('Cost History')


# Plot BC loss vs epochs
plt.plot(BC_loss)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('BC Loss')
plt.title('Cost History')


# Plot total loss, PDE loss and BC loss versus epochs
plt.plot(BC_loss,label = 'BC loss')
plt.plot(PDE_loss,label = 'PDE loss')
plt.plot(Total_loss,label = 'Total loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost History')
plt.legend()


# ## 4. Save the trained model (parameters) using ‘torch.save()’

torch.save(model.state_dict(), 'model_2000.pth')


# ### As we can see above plots were fairly unsatisfactory. We need more epochs to get better results. Let us train our data for 10000 epochs


extra_epochs = 8000


for epoch in range(extra_epochs):
    train(epoch+2001)


# ## 1. Plot the displacement field contours in 𝑥− and 𝑦− directions after training the model.


ui_preds = model.forward(xy_f_train)
ub_preds = model.forward(xy_b_train)
u_preds = torch.cat((ui_preds, ub_preds), dim = 0).detach().numpy()
u_plot = u_preds[:,0]
v_plot = u_preds[:,1]
xy_all = torch.cat((xy_f_train, xy_b_train), dim=0).detach().numpy()
x = xy_all[:,0]
y = xy_all[:,1]    
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Interpolate the data onto the grid
U = griddata((x, y), u_plot, (X, Y), method='linear')
V = griddata((x, y), v_plot, (X, Y), method='linear')

# Plot the results
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
im1 = ax1.imshow(U, cmap='rainbow', origin='lower')
im2 = ax2.imshow(V, cmap='rainbow', origin='lower')
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

ax1.set_title('Displacement in x direction')
ax2.set_title('Displacement in y direction')
plt.show()


# ## 2. Plot the total loss, PDE loss and BC loss versus epochs.


plt.plot(Total_loss)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cost History')



plt.plot(PDE_loss)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('PDE Loss')
plt.title('Cost History')



plt.plot(BC_loss)
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('BC Loss')
plt.title('Cost History')



plt.plot(BC_loss,label = 'BC loss')
plt.plot(PDE_loss,label = 'PDE loss')
plt.plot(Total_loss,label = 'Total loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost History')
plt.legend()


# ## 3. Save the trained model (parameters) using ‘torch.save()’


torch.save(model.state_dict(), 'model_10000.pth')



# # Solution of 2D Forward Problem of Linear Elasticity for Plane Stress Boundary Value Problem using Physics-Informed Neural Networks (PINN)


import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math

torch.manual_seed(123456)
np.random.seed(123456)


# ## 4. Generate random collocation points sampled from a uniform distribution in the interior (n = 2000 pairs) and boundary of the domain (n = 4*100 pairs). Concatenate all the sampled points and give as input to the trained model.
# 


nb = 100

x_0 = torch.rand(100, requires_grad=True).unsqueeze(dim=1)
x_0 = torch.cat((x_0, torch.zeros_like(x_0)), dim = 1)

x_1 = torch.rand(100, requires_grad=True).unsqueeze(dim=1)
x_1 = torch.cat((x_1, torch.ones_like(x_1)), dim = 1)

y_0 = torch.rand(100, requires_grad=True).unsqueeze(dim=1)
y_0 = torch.cat((torch.zeros_like(y_0),y_0), dim = 1)

y_1 = torch.rand(100, requires_grad=True).unsqueeze(dim=1)
y_1 = torch.cat((torch.ones_like(y_1),y_1), dim = 1)

# Generate boundary points
boundary_test = torch.cat((x_0,x_1,y_0,y_1),dim=0)

# Generate internal points
ni = 2000
interior_test = torch.rand(ni,2, requires_grad = True)


# ## 5. Load the saved model using ‘torch.load()’
# 



import torch.nn as nn
## Need architecture to load parameters using torch.load()
class Multiclass(nn.Module):
    def __init__(self):
        super(Multiclass, self).__init__()
        # Define your model here (refer: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
        self.Nnet = nn.Sequential()
        self.Nnet.add_module('Hidden_layer_1', nn.Linear(2,30))    # First linear layer
        self.Nnet.add_module('Tanh_layer_1', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_2', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_2', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_3', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_3', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_4', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_4', nn.Tanh())
        self.Nnet.add_module('Hidden_layer_5', nn.Linear(30,30))
        self.Nnet.add_module('Tanh_layer_5', nn.Tanh())
        self.Nnet.add_module('Output Layer', nn.Linear(30,2))      
        print(self.Nnet)                                        # Print model summary

    # Forward Feed
    def forward(self, x):
        y = self.Nnet(x)
        return y
    



# Load the saved model

model = Multiclass()
model.load_state_dict(torch.load("model_2000.pth"))


# ## 6. Test your trained model on the generated collocation points and plot the displacement field.


ui_preds = model.forward(interior_test)
ub_preds = model.forward(boundary_test)
u_preds = torch.cat((ui_preds, ub_preds), dim = 0).detach().numpy()
u_plot = u_preds[:,0]
v_plot = u_preds[:,1]
xy_all = torch.cat((interior_test, boundary_test), dim=0).detach().numpy()
x = xy_all[:,0]
y = xy_all[:,1]    
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Interpolate the data onto the grid
U = griddata((x, y), u_plot, (X, Y), method='linear')
V = griddata((x, y), v_plot, (X, Y), method='linear')

# Plot the results
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
im1 = ax1.imshow(U, cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()],origin='lower')
im2 = ax2.imshow(V, cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()],origin='lower')
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

ax1.set_title('Displacement in x direction')
ax2.set_title('Displacement in y direction')
plt.show()


# # Model parameters trained after 10000 epochs


# Load the saved model

model2 = Multiclass()
model2.load_state_dict(torch.load("model_10000.pth"))

ui_preds = model2.forward(interior_test)
ub_preds = model2.forward(boundary_test)
u_preds = torch.cat((ui_preds, ub_preds), dim = 0).detach().numpy()
u_plot = u_preds[:,0]
v_plot = u_preds[:,1]
xy_all = torch.cat((interior_test, boundary_test), dim=0).detach().numpy()
x = xy_all[:,0]
y = xy_all[:,1]    
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

# Interpolate the data onto the grid
U = griddata((x, y), u_plot, (X, Y), method='linear')
V = griddata((x, y), v_plot, (X, Y), method='linear')

# Plot the results
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))
im1 = ax1.imshow(U, cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
im2 = ax2.imshow(V, cmap='rainbow', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
fig.colorbar(im1, ax=ax1)
fig.colorbar(im2, ax=ax2)

ax1.set_title('Displacement in x direction')
ax2.set_title('Displacement in y direction')
plt.show()









