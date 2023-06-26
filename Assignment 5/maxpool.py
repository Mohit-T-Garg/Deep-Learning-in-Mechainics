import numpy as np

class MaxPool2:
  # A Max Pooling layer using a pool size of 2.

  def iterate_regions(self, image):
    '''
    Generates non-overlapping 2x2 image regions to pool over.
    - image is a 2d numpy array
    hint: You can use 'yield' statement in Python to create iterate regions
    '''
    h, w, _ = image.shape
    h1,w1 = h//2 , w//2

    for i in range(h1):
      for j in range(w1):
        im_region = image[(i*2):(i*2 + 2), (j*2):(j*2 + 2)]
        yield im_region, i, j
    
    
    

  def forward(self, input):
    '''
    Performs a forward pass of the maxpool layer using the given input.
    Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
    - input is a 3d numpy array with dimensions (h, w, num_filters)
    '''
    h, w, num_filters = input.shape
    h1,w1 = h//2 , w//2
    self.last_input = input
    output = np.zeros(( h1, w1, num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.amax(im_region, axis=(0, 1))
    
    return output
    
  def backprop(self, d_L_d_out):
    '''
    Performs a backward pass of the maxpool layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    '''
    d_L_d_input = np.zeros(self.last_input.shape)

    for im_region, i, j in self.iterate_regions(self.last_input):
      h, w, f = im_region.shape
      amax = np.amax(im_region, axis=(0, 1))

    # Iterate over every pixel of 'im_region' and check whether pixel value
    # matches one of the elements of amax. If true, copy the gradient to that location.
      for a in range(h):
        for b in range(w):
          for c in range(f):
            if im_region[a, b, c] == amax[c]:
                d_L_d_input[2*i + a, 2*j + b, c] = d_L_d_out[a, b, c]


    return d_L_d_input
