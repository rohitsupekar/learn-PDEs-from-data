"""
Contains sparsity promoting solvers
1. STRidge
2. IHTd
3. LASSO (to be implemented)
"""
from sklearn import preprocessing
import numpy as np

def STRidge(X, y, lam, tol, maxit=1000, W=None, standardize = False, print_flag = False, \
            thresh_nonzero = None):
    """
    Sequential Threshold Ridge Regression algorithm.
    NOTE: this assumes y is single column
    thresh_nonzero: vector which is the same length as columns in X0 which has
    0 where the feature is not to be thresholded and 1 where it is thresholded
    """

    n,d = X.shape

    #Data standardiation: important for ridge regression because the penalty on the coefficients is uniform
    #Make columns of X to be zero mean and unit variance
    #Make y data to be zero mean (but not unit variance)
    if standardize:
        X_std = preprocessing.scale(X)
        y_std = preprocessing.scale(y, with_std=False)
    else:
        X_std = X
        y_std = y

    #set default weights matrix
    if W is None:
        W = np.ones((d,1))
    else:
        #keep only the diagonal of the weights Matrix
        W = np.diagonal(W).reshape(d,1)

    #set default threshold vector (all terms one if all the coefficients are to be included
    # in the thresholding)
    if thresh_nonzero is None:
        thresh_nonzero = np.ones((d,1))

    # Get the standard ridge esitmate
    w = np.linalg.lstsq(X_std.T@X_std + lam*np.diag(W)@np.diag(W), X_std.T@y_std, rcond=-1)[0]
    num_relevant = d

    # Thresholding loop
    for j in range(maxit):

        if print_flag:
            print("iter, num_relev = ", j, num_relevant)

        # Figure out which items to cut out
        smallinds = np.where(abs(w) - tol*thresh_nonzero < 0)[0]
        biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(biginds):
            if print_flag: print("breaking")
            break
        else:
            num_relevant = len(biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(biginds) == 0:
            if j == 0:
                if print_flag: print("All coeffs < tolerance at 1st iteration!")
                return np.zeros((d,1))
            else:
                if print_flag: print("All coeffs < tolerance at %i iteration!" %(j))
                #break -- this statement will just keep coefficients at previous iteration
                return np.zeros((d,1)) # -- this statement will just return zeros

        # New guess
        w[smallinds] = 0
        w[biginds] = np.linalg.lstsq(X_std[:, biginds].T@X_std[:, biginds] + lam*np.diag(W[biginds])@np.diag(W[biginds]),X_std[:, biginds].T@y_std, rcond=-1)[0]


    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y, rcond=-1)[0]

    return w


def IHTd(X, y, lasso_lam, max_iter, sub_iter, tol, htp_flag, print_flag=False):
    n,d = X.shape
    coeff_old  = np.zeros((d,1))
    coeff_new  = np.zeros((d,1))
    L = 1.0
    LL = np.linalg.norm(X.T.dot(X),2)
    one  =  (1./L)*(X.T.dot(y))
    XTX  = X.T.dot(X)

    coeff_old =  one #+ 0.01*np.random.normal(0, 1, d)
    support =  np.arange(0,d)
    frac_iter = 0
    time_step = (1.0/LL)
    for iteration in range(0, max_iter):
        if print_flag:
            print('iter = %i' %(iteration))
        temp = np.zeros((d,1))
        if(htp_flag == 1):
            temp = coeff_old + (2.0/LL) *(one  - XTX.dot(coeff_old)) #inclusion will make similar algorithm to HTP
        else:
            temp = coeff_old.copy()

        smallinds   = np.where( abs(temp) <= lasso_lam/(L))[0]
        biginds     = [i for i in range(d) if i not in smallinds]
        if( len(biginds) == 0 ):  # if( len(biginds)==0 and iteration !=0 ):
            coeff_new[:] = 0
            return coeff_new
        else:
            #temp    = np.zeros((1,d)) # added this
            temper  = np.zeros((len(biginds), 1))
            grad    = np.zeros((len(biginds),1))
            grad_A  = np.zeros((n,1))
            LLL     = np.linalg.norm(X[:, biginds].T.dot(X[:, biginds]), 2)
            XTy_s   = X[:, biginds].T.dot(y)
            XTX_s   = X[:, biginds].T.dot( X[:, biginds])
            for k in range(0, sub_iter):
                temp_2      = XTX_s.dot( temp[biginds,:] )
                grad        = (XTy_s - temp_2)
                grad_A      = X[:, biginds].dot(grad)
                if(grad.T.dot(grad) == 0):
                    break
                else:
                    time_step = (grad.T.dot(grad))/(grad_A.T.dot(grad_A))

                temper = temp[biginds] + (time_step)*grad
                if( np.linalg.norm( y - X[:, biginds].dot(temper), 2)**2 < (len(biginds)*lasso_lam) ):
                    coeff_new[biginds]   = temper.copy()
                    coeff_new[smallinds] = 0
                    frac_iter = frac_iter + 1
                    break

                temp[biginds]   = temper.copy() # copy to old

            coeff_new[biginds]   = temper.copy() # new coeffs value
            coeff_new[smallinds] = 0                  # new coeffs value

        # check for convergence
        if( (np.linalg.norm(coeff_new - coeff_old, np.inf) < tol) and np.all(support == biginds) ):
            if(print_flag):
                print(" converged at iteration ", iteration,  " frac_iter ", frac_iter/(iteration+1), " flag ", htp_flag)
            return coeff_new

        # copy the values between old and new
        support = biginds.copy()
        coeff_old = coeff_new.copy()

    return coeff_new #return a column vector
