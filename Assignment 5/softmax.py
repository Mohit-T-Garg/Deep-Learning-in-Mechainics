import numpy as np

class Softmax:
  # A standard fully-connected layer with softmax activation.

  def __init__(self, input_len, output_len):
    # After randomly intializing the weights, divide by input_len to reduce the variance
    self.weights =  np.ones((input_len, output_len)) / input_len
    self.biases = np.zeros(output_len)

  def forward(self, input):
    '''
    Performs a forward pass of the softmax layer using the given input.
    Returns a 1d numpy array containing the respective probability values.
    - input can be any array with any dimensions.
    '''
    self.last_input_shape = input.shape
    self.last_input = input.flatten()
    #print(input.shape,self.weights.shape)
    self.last_totals = np.dot(self.last_input,self.weights) + self.biases

    prob = np.exp(self.last_totals)
    
    return prob/np.sum(prob,axis=0)
    
  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    - Only 1 element of d_L_d_out will be nonzero.
    '''

    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue  # Skips the current iteration of the loop

      # e^totals (Evaluate exponents for the values passed into softmax activation function)
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      Dt_Dw = d_t_d_w[np.newaxis]
      DL_Dt = d_L_d_t[np.newaxis]
      d_L_d_w = Dt_Dw.T @ DL_Dt
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t

      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
