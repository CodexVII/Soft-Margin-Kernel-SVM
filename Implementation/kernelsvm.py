##--------------------------------------------------------------------------
##
##  kernelsvm.py
##
##  Routines to generate a hard-margin kernel-based SVM.  The solution to
##  the dual form Lagrangian optimization problem:
##
##      Maximize:
##
##        W(L) = \sum_i L_i - 1/2 sum_i sum_j L_i L_j t_i t_j K(x_i, x_j)
##
##      subject to:  \sum_i t_i L_i  =  0   and L_i >= 0.
##
##  (where L is a vector of n Lagrange multipliers and K is a kernel
##  function taking two vectors as arguments and returning a scalar) is
##  found using the quadratic program solver "qp" from the convex
##  optimization package "cvxopt" (which has to be installed).
##  
##  Example run, generate weights and bias for a 2-input AND binary
##  classifier.
##
##  >>> Xs = makeBinarySequence(2)   ## Generate list [[0,0],[0,1]...]
##  >>> Ts = [-1,-1,-1,+1]           ## Desired response for a binary AND.
##  >>> stat,Ls = makeLambdas(xs,ts) ## Solve W(Ls) for Lagrange mults.
##                                   ## N.B., status == 'optimal' if a
##                                   ## solution has been found.
##  >>> b = makeB(Xs,Ts,Ls)          ## Find bias.
##  >>> classify([0,1],Xs,Ts,Ls,b)   ## Test classification.
##  >>> testClassifier(Xs,Ts,Ls,b)   ## Exhaustive test on all training
##                                   ## patterns.  See documentation below.
##  >>> plotContours(Xs,Ts,Ls,b)     ## Plot the decision boundary and the
##                                   ## +ve/-ve margins for a 2-d 
##                                   ## classification problem.
##
##  N.B., "makeLambdas", "makeB" and "classify" all have an optional
##  parameter, the kernel function K.  This defaults to a polynomial
##  kernel (x.y + 1.0)^2.  Any  function that accepts two vectors as 
##  arguments and returns a scalar can be used here.  If you need to
##  specify an alternative kernel, use keyword K (see 3-input XOR
##  example below, which uses a cubic kernel).  Definitions for 
##  polynomial, rbf and linear kernels are provided as examples.
##  
##
##--------------------------------------------------------------------------
##
##  Routines
##
##      makeLambdas  -- Generate the n Lagrange multipliers that
##                      represent the maximum-point of the dual
##                      optimization problem W(L).
##                      N.B., maximizing W(L) is a quadratic convex
##                      optimization problem, so the "qp" solver from
##                      "cvxopt" actaully does the work.  Most of what
##                      this routine does is simply setting up the
##                      arguments for the call to "qp".
##
##              Arguments:
##
##                 Xs -- A list of vectors (lists) representing
##                       the input training patterns for this
##                       problem.
##
##                       e.g., [[0,0],[0,1],[1,0],[1,1]], the set
##                       of binary training patterns.
##
##                 Ts -- A list of desired outputs.  These must
##                       be values in the set {-1,+1}.  If there
##                       are n input vectors in x, there must be
##                       exactly n values in t.
##
##                       e.g., [-1,-1,-1,1] -- the outputs
##                       for a 2-input AND function.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##
##              Returns:
##
##                 A 2-tuple (status,Ls).
##                 1) status:  The first element of the tuple is the 
##                    return status of the "qp" solver.  This will be the 
##                    string "optimal" if a solution has been found.  If 
##                    no solution can be found (say if an XOR problem is
##                    presented to the solver), status typically comes back
##                    as "unknown".
##                 2) Ls:  The second element of the tuple is a list of  
##                    the n Lagrange multipliers.  Element 0 of this list
##                    is the first multiplier and corresponds to Xs[0] and
##                    Ts[0], element 1 is the multiplier corresponding to 
##                    Xs[1] and Ts[1], etc.  These values are only 
##                    meaningful if the first element of the tuple, status
##                    has returned as "optimal".
##
##      --------------------------------------------------------------------
##
##      makeB  -- Given the set of training vectors, "Xs", the set of
##                training responses "Ts", and the set of Lagrange
##                multipliers for the problem "Ls", return the bias for
##                the classifier.
##
##              Arguments:
##
##                 Xs -- Inputs, as "Xs" in makeLambdas.
##
##                 Ts -- A list of desired outputs.  As "Ts" in
##                       makeLambdas.
##
##                 Ls -- A list of Lagrange multipliers, the
##                       solution to the contrained optimaztion
##                       of W(L) as returned by a call to
##                       makeLambdas.  N.B., if this argument is
##                       None (the default), this routine will call
##                       generateLambdas automatically.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##
##              Returns:
##
##                 A double, the bias, "b".
##
##      --------------------------------------------------------------------
##
##      classify -- Classify an input vector using the Lagrange
##                  multipliers for the problem, the set of training
##                  inputs, "Xs", the set of desired outputs, "Ts" and 
##                  bias "b", classify a vector "x".
##
##              Arguments:
##
##                 x -- An input vector to classify (a list of values).
##
##                 Xs -- A list of the training input vectors (hence a list
##                       of n-element lists).
##
##                 Ts -- A list of desired outputs.  As "Ts" in
##                       makeLambdas.
##
##                 Ls -- A list of Lagrange multipliers.
##
##                 b  -- The classifier bias, as generated by makeB.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##                 verbose -- Controls whether or not the routine
##                       prints details about the current classification to
##                       the terminal as well as returning a status
##                       value.  Defaults to True.
##
##              Returns:
##
##                 A classification, +1, -1 or 0 (which indicates an
##                 error in the classification, and shouldn't happen).
##
##      --------------------------------------------------------------------
##
##      testClassifier(Xs,Ts,Ls,b,K,verbose) --
##                  Test a classifier by checking to see if its response
##                  to every training input Xs[i] is the desired output
##                  Ts[i].
##
##              Arguments:
##
##                 Xs -- A list of vectors (lists) representing
##                       the input training patterns for this
##                       problem.
##
##                       e.g., [[0,0],[0,1],[1,0],[1,1]], the set
##                       of binary training patterns.
##
##                 Ts -- A list of desired outputs.  These must
##                       be values in the set {-1,+1}.  If there
##                       are n input vectors in x, there must be
##                       exactly n values in t.
##
##                       e.g., [-1,-1,-1,1] -- the outputs
##                       for a 2-input AND function.
##
##                 Ls -- A list of Lagrange multipliers.
##
##                 b  -- The classifier bias, as generated by makeB.
##
##                 K  -- A Kernel function.  This should be a
##                       function taking two vectors as arguments
##                       and returning a scalar.  This is an optional
##                       parameter, if omitted a default polynomial
##                       kernel kernel (x.y + 1.0)^2 is used.
##
##                 verbose -- Controls whether or not the routine
##                       prints details of misclassifications to the
##                       terminal as well as returning a status
##                       value.  Defaults to True.
##
##
##              Returns:
##
##                 True/False
##
##      --------------------------------------------------------------------
##
##      rbfKernel(x,y,s2) --
##                  Radial basis function kernel exp(-||x-y||^2/2*sigma^2).
##
##              Arguments:
##
##                 x,y  -- n-element vectors.
##
##                 s2   -- Variance of the R.B.F. kernel.  Squared standard
##                         deviation, sigma^2.
##
##              Returns:
##
##                 Scalar value of kernel for the 2 input vectors.
##
##
##  ------------------------------------------------------------------------
##
##  Support routines
##
##
##      makeBinarySequence -- Generate a list of the 2^d d-element "vectors"
##                            comprising the complete binary sequence in
##                            d bits.
##
##              Argument:
##
##                 d -- Length of each vector in the output list.  Defaults
##                      to 2 (which will generate the 4 "vector" list
##                      [[0,0], [0,1], [1,0], [1,1]].
##
##              Returns:
##
##                 A list of 2^d elements, each of which is one vector in
##                 the binary sequence.  Example, a call with d=3 will
##                 return the 8-element list [[0,0,0], [0,0,1], [0,1,0],
##                 [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]].
##
##      
##      --------------------------------------------------------------------
##
##      makeP -- Generates the P matrix for a kernel SVM problem.  See the
##               comments associated with generateLambdas below for a
##               discussion of the form and role of the P matrix.
##
##
##--------------------------------------------------------------------------
##


from cvxopt import matrix,solvers
from math import exp
import numpy as np
from numpy import array


def rbfKernel(v1,v2,sigma2=1.0):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x,y: (x-y)*(x-y), v1,v2))  ## Squared mag of diff.
    return exp(-mag2/(2.0*sigma2))


def makeLambdas(Xs,Ts,C,K=rbfKernel):
    "Solve constrained maximaization problem and return list of l's."
    P = makeP(Xs,Ts,K)            ## Build the P matrix.
    n = len(Ts)
    q = matrix(-1.0,(n,1))        ## This builds an n-element column 
                                  ## vector of -1.0's (note the double-
                                  ## precision constant).
    
    ## 2n x n column vector where the first n contains 0's while the
    ## second half contains C    
    h = matrix(0.0,(2*n,1))       
    for i in range((len(h)/2), len(h)):
        h[i]=C

    ## Stack two n x n matrices on top of one another
    ## Top matrix contains diagonal of -1's and bottom contains 1's 
    G1 = matrix(0.0,(n,n))        
    G1[::(n+1)] = -1.0                                               
    G2 = matrix(0.0,(n,n))        
    G2[::(n+1)] = 1.0             

    G_a = array(G1)
    G_b = array(G2)
    GG = np.vstack((G_a,G_b))
    
    G = matrix(GG)
    
    A = matrix(Ts,(1,n),tc='d')   ## A is an n-element row vector of 
                                  ## training outputs.
      
    r = solvers.qp(P,q,G,h,A,matrix(0.0))  ## "qp" returns a dict, r.
    
    Ls = [round(l,6) for l in list(r['x'])] ## "L's" are under the 'x' key.
    return (r['status'],Ls)



def makeB(Xs,Ts,C=1,Ls=None,K=rbfKernel):
    "Generate the bias given Xs, Ts and (optionally) Ls and K"
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    sv_count = 0
    b_sum = 0.0
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10 and Ls[n] < C:   ## 1e-10 for numerical stability.
            sv_count += 1
            b_sum += Ts[n]
            for i in range(len(Ts)):
                if Ls[i] >= 1e-10:
                    b_sum -= Ls[i] * Ts[i] * K(Xs[i],Xs[n])
            
    return b_sum/sv_count


def classify(x,Xs,Ts,C=1,Ls=None,b=None,K=rbfKernel,verbose=True):
    "Classify an input x into {-1,+1} given support vectors, outputs and L." 
    ## No Lagrange multipliers supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:  b = makeB(Xs,Ts,C,Ls,K)
    ## Do classification.  y is the "activation level".
    y = b
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:
            y += Ls[n] * Ts[n] * K(Xs[n],x)

    if verbose:
        print "%s %8.5f  --> " %(x, y),
        if y > 0.0: print "+1"
        elif y < 0.0: print "-1"
        else: print "0  (ERROR)"
    if y > 0.0: return +1
    elif y < 0.0: return -1
    else: return 0 


def testClassifier(Xs,Ts,C=1,Ls=None,b=None,K=rbfKernel,verbose=True):
    "Test a classifier specifed by Lagrange mults, bias and kernel on all Xs/Ts pairs."
    assert len(Xs) == len(Ts)
    ## No Ls supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts,C,K)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
        print "Lagrange multipliers:",Ls
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b == None:
        b = makeB(Xs,Ts,C,Ls,K)
        print "Bias:",b
    ## Do classification test.
    good = True
    missed = 0
    fn = 0
    fp = 0
    for i in range(len(Xs)):
        c = classify(Xs[i],Xs,Ts,C,Ls,b,K=K)
        if c != Ts[i]:
            missed += 1
            if c == 1 and Ts[i] == -1:
                fp += 1
            elif c == -1 and Ts[i] == 1:
                fn += 1
            if verbose:
                print "Misclassification: input %s, output %d, expected %d" %\
                      (Xs[i],c,Ts[i])
            good = False
    return good, missed, fn, fp


##--------------------------------------------------------------------------
##
##  Auxiliary routines.
##
##--------------------------------------------------------------------------
##

## Make the P matrix for a nonlinear, kernel-based SVM problem.
##
def makeP(xs,ts,K):
    """Make the P matrix given the list of training vectors,
       desired outputs and kernel."""
    N = len(xs)
    assert N == len(ts)
    P = matrix(0.0,(N,N),tc='d')
    for i in range(N):
        for j in range(N):
            P[i,j] = ts[i] * ts[j] * K(xs[i],xs[j])
    return P

## Radial Basis Function with sigma^2=0.15
##
def rbf2(x,y):
    return rbfKernel(x,y,0.15)


##--------------------------------------------------------------------------
##
##  Working with the trainng set
##
##--------------------------------------------------------------------------
##

## Open up the file containing the training set and place each column 
## in a suitable list.
#
x1 = []     ## stores x
x2 = []     ## stores y
Ts = []     ## desired output of training set
with open('training-dataset-aut-2016.txt') as f:
    for line in f:
        data = line.split()
        x1.append(float(data[0]))
        x2.append(float(data[1]))
        Ts.append(int(data[2]))
        

## Place x1 and x2 into the single Xs list
num_points = len(x1)
Xs = num_points*[None]     ## training set
for i in range(num_points):
    Xs[i]=[x1[i],x2[i]]
    
## Test training set with C=1 sigma^2=0.15
#
C1 = 1
print "\n\nAttempting to generate LMs for training test using rbf kernel with C=1 sigma^2=0.15"
status,Ls1=makeLambdas(Xs,Ts,C1,K=rbf2)
if status == 'optimal':
    b=makeB(Xs,Ts,C1,Ls1,K=rbf2)
    passed,missed,fn,fp = testClassifier(Xs,Ts,C1,Ls1,b,K=rbf2)
    accuracy=((float(num_points)-missed)/float(num_points))*100.0
    if passed:
        print "  Check PASSED"
    else:
        print "  Check FAILED: Classifier does not work corectly on training inputs"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)
    
# Test training set with C=1e6 sigma^2=0.15

C2 = 1e6
print "\n\nAttempting to generate LMs for training test using rbf kernel with C=1e6 sigma^2=0.15"
status,Ls2=makeLambdas(Xs,Ts,C2,K=rbf2)
if status == 'optimal':
    b=makeB(Xs,Ts,C2,Ls2,K=rbf2)
    passed,missed,fn,fp = testClassifier(Xs,Ts,C2,Ls2,b,K=rbf2)
    accuracy=((float(num_points)-missed)/float(num_points))*100.0
    if passed:
        print "  Check PASSED"
    else:
        print "  Check FAILED: Classifier does not work corectly on training inputs"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)

##--------------------------------------------------------------------------
##
##  Working with the testing set
##  Reuse the corresponding L's and C's from the training phase
##
##--------------------------------------------------------------------------
##
x1 = []     ## stores x
x2 = []     ## stores y
Ts = []     ## desired output of training set
with open('testing-dataset-aut-2016.txt') as f:
    for line in f:
        data = line.split()
        x1.append(float(data[0]))
        x2.append(float(data[1]))
        Ts.append(int(data[2]))
        

## Place x1 and x2 into the single Xs list
num_points = len(x1)
Xs = num_points*[None]     ## training set
for i in range(num_points):
    Xs[i]=[x1[i],x2[i]]
    
## Test set with C=1 sigma^2=0.15
#
print "\n\nAttempting to generate LMs for test set using rbf kernel with C=1 sigma^2=0.15"
passed,missed,fn,fp = testClassifier(Xs,Ts,C1,Ls1,b,K=rbf2)
accuracy=((float(num_points)-missed)/float(num_points))*100.0
if passed:
    print "  Check PASSED"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)

    
## Test set with C=1e6 sigma^2=0.15
#
print "\n\nAttempting to generate LMs for test set using rbf kernel with C=1e6 sigma^2=0.15"
passed,missed,fn,fp = testClassifier(Xs,Ts,C2,Ls2,b,K=rbf2)
accuracy=((float(num_points)-missed)/float(num_points))*100.0
if passed:
    print "  Check PASSED"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)