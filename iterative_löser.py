import numpy as np

# A function jacobi , which solves an LGS using the Jacobi method

def jacobi(A , b , X0 = None , tol = 1.e-8 , max_n = 100):
    
    (m,n)= A.shape
    
    #Initialising number of iterations
    k = 1
    #Define starting vector 
    if X0 == None:
        X0 = np.zeros(shape = n)
        nIt = 100
        x = X0.copy()
        
        #Loop to calculate jacobi
        
    while k <= max_n :
        for i in range(n):
            s = 0
            for j in range(n):
                if(i == j):
                    continue
                else:
                        s += A[i][j] * x[j]
            X0[i] = (-s + b[i]) / (A[i][i])
            #Calculates error , here I used residual norm (x_currentstep-x_previousstep)
            err = np.linalg.norm(X0 - x)
            #Checks error condition
            if (err < tol):
                return X0 , nIt
            #Incrementing k
            k += 1
            #Updating parameter
            for i in range(n):
                x[i] = X0[i]
                
def test_jacobi(): 
    A = np.array([[4 ,-1 , 1] , [-1 , 4 , -2] , [1 , -2 , 4]])
    b = np.array([[12] , [-1] , [5]])
    x = np.ones(shape = 3)
    
    (x,its) = jacobi(A , b , tol = 1.e-8 )
    
    print ('jacobi Lösüng nach {} Iterationsschritten : {}' .format (its,x))
    

# A function gauss_seidel , which solves an LGS using the Gauss-Seidel method  
 
def gauss_seidel (A , b , X0 = None , tol = 1.e-8 , max_n = 100):
    
    (m,n) = A.shape 
    
    # Initialising number of iteration
    k = 1
    #Define starting vector 
    if X0 == None:
        X0 = np.zeros(shape = n)
        nIt = 100
        x = X0.copy()
        
        #Loop to calculate Gauss_Seidel
    while k <= max_n :
        for i in range (n):
            s = 0
            for j in range (i):
                if (i==j):
                    continue
                else:
                    s += A[i][j]*X0[j]
            for j in range (i + 1 , n):
                if (i==j):
                    continue
                else:
                    s += A[i][j]*x[j]
            X0[i] = (-s + b[i]) / (A[i][i])
            #Calculates error , here I used residual norm (x_currentstep-x_previousstep)
            err = np.linalg.norm(X0 - x)
            #Checks error condition
            if (err < tol):
                return X0 , nIt
            #Incrementing k
            k += 1
            #Updating parameter
            for i in range(n):
                x[i] = X0[i]
                
def test_gauss_seidel():
    A = np.array([[4 ,-1 , 1] , [-1 , 4 , -2] , [1 , -2 , 4]])
    b = np.array([[12] , [-1] , [5]])
    x = np.ones(shape = 3)
    
    (x,its) = gauss_seidel (A , b , tol = 1.e-8 )
    
    print ('gauss_seidel Lösüng nach {} Iterationsschritten : {}' .format (its,x))
    

# A function sor , which solves an LGS using the SOR method

def sor (A , b , X0 = None , tol = 1.e-8 , max_n = 100, w = 1.0):
    
    (m,n) = A.shape 
    
    # Initialising number of iteration
    k = 1
    #Define starting vector 
    if X0 == None:
        X0 = np.zeros(shape = n)
        nIt = 100
        x = X0.copy()
        #omega
        w = 1.0
        
        #Loop to calculate sor
    while k <= max_n :
        for i in range (n):
            s = 0
            for j in range (i):
                if (i==j):
                    continue
                else:
                    s += A[i][j]*X0[j]
            for j in range (i + 1 , n):
                if (i==j):
                    continue
                else:
                    s += A[i][j]*x[j]
                    
            # SOR implementation
            X0[i] = (1-w) * X0[i] + (w / A[i][i]) * (b[i]-s)
            
            #Calculates error , here I used residual norm (x_currentstep-x_previousstep)
            err = np.linalg.norm(X0 - x)
            #Checks error condition
            if (err < tol):
                return x , nIt
            #Incrementing k
            k += 1
            #Updating parameter
            for i in range(n):
                x[i] = X0[i]

def test_sor():
    
    A = np.array([[4 ,-1 , 1] , [-1 , 4 , -2] , [1 , -2 , 4]])
    b = np.array([[12] , [-1] , [5]])
    x = np.ones(shape = 3)
    (x,its) = sor (A, b, tol=1.e-8, w = 1.0)
    
    print ('sor (für omega = 1.) Lösüng nach {} Iterationsschritten : {}' .format (its,x))
    
    
    
if __name__ == "__main__":
    
    test_jacobi()
    test_gauss_seidel()
    test_sor()