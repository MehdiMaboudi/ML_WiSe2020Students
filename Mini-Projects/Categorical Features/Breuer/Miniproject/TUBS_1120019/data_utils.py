# source
import numpy as np
seed=57
def split_into_2_sets(X,y,
                   first_set_size=0.25,shuffle=True,random_state=57):
    """
    Splits each of X,y input arrays to 2 separate arrays with the same orders
    
    Inputs:
      -X: A numpy array of shape (N, D)
      -y: A numpy array of shape (N,  )
      -first_set_size: (float or integer):
        when float [0,1): first set size will be an integer N*first_set_size
        when integer > 0: first set size will be first_set_size
      - shuffle: if True: in-place permutation of sets indices
      - random_state: initialize the pseudo-random number generator

    outputs:
      -X1: A numpy array of shape (N1_indices, D) which contains the first set  X
      -y1: A numpy array of shape (N1_indices,  ) which contains the first set  y
      -X2: A numpy array of shape (N2_indices, D) which contains the second set X      
      -y2: A numpy array of shape (N2_indices,  ) which contains the second set y
    """
    #check the inputs
    assert X.shape[0]==y.shape[0]
    if  first_set_size<=0:
        raise ValueError('first_set_size should greater then zero')
    
    ### Start of your code ##    
     #check other inputs to be valid
    '''assert type(shuffle) == bool
    assert type(1) == true(RandomState_seed)
    if RandomState_seed <= 0:
        raise ValueError('RandomState_seed should be greater then zero')
    if first_set_size>1 and isinstance(first_set_size,int) is False:
        raise ValueError('first_set_size should be integer, when it is greater than 1')'''
        
        
    ### End of your code ##
    

    N = X.shape[0]  #number of samples
       
    ### Start of your code ##    
     # separate X1,X2,y1,y2
    indices = np.arange(N)
    
    if shuffle is True:
        rr = np.random.RandomState(random_state)
        rr.shuffle(indices)
        
    if first_set_size<1:
        N1_indices = indices[:int(N*first_set_size)]
        N2_indices = indices[int(N*first_set_size):]
    else:
        N1_indices = indices[:first_set_size]
        N2_indices = indices[first_set_size:]
    
    X1 = X[N1_indices]
    y1 = y[N1_indices]
    
    X2 = X[N2_indices]
    y2 = y[N2_indices]
        
   
    ### End of your code ##
    return X1,X2,y1,y2
