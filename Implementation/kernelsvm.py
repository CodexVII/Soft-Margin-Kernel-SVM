import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import savefig
from math import exp
from numpy import array
from cvxopt import matrix,solvers

def rbfKernel(v1,v2,sigma2=1.0):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x,y: (x-y)*(x-y), v1,v2))  ## Squared mag of diff.
    return exp(-mag2/(2.0*sigma2))

## Radial Basis Function with sigma^2=0.15
##
def rbf2(x,y):
    return rbfKernel(x,y,0.15)



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
                if Ls[i] >= 1e-10 and Ls[n] < C:
                    b_sum -= Ls[i] * Ts[i] * K(Xs[i],Xs[n])
            
    return b_sum/sv_count


def classify(x,Xs,Ts,C=1,Ls=None,b=None,K=rbf2,verbose=True):
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
        c = classify(Xs[i],Xs,Ts,C,Ls,b,K=K)    #possibly needs to be modified
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
    
## Training set with C=1 sigma^2=0.15
#
C1 = 1
print "\n\nAttempting to generate LMs for training test using rbf kernel with C=1 sigma^2=0.15"
status,Ls1=makeLambdas(Xs,Ts,C1,K=rbf2)
if status == 'optimal':
    b1=makeB(Xs,Ts,C1,Ls1,K=rbf2)
    passed,missed,fn,fp = testClassifier(Xs,Ts,C1,Ls1,b1,K=rbf2)
    accuracy=((float(num_points)-missed)/float(num_points))*100.0
    if passed:
        print "  Check PASSED"
    else:
        print "  Check FAILED: Classifier does not work corectly on training inputs"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)
    
# Training set with C=1e6 sigma^2=0.15

C2 = 1e6
print "\n\nAttempting to generate LMs for training test using rbf kernel with C=1e6 sigma^2=0.15"
status,Ls2=makeLambdas(Xs,Ts,C2,K=rbf2)
if status == 'optimal':
    b2=makeB(Xs,Ts,C2,Ls2,K=rbf2)
    passed,missed,fn,fp = testClassifier(Xs,Ts,C2,Ls2,b2,K=rbf2)
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
x1_t = []     ## stores x
x2_t = []     ## stores y
Ts = []     ## desired output of training set
with open('testing-dataset-aut-2016.txt') as f:
    for line in f:
        data = line.split()
        x1_t.append(float(data[0]))
        x2_t.append(float(data[1]))
        Ts.append(int(data[2]))
        

## Place x1 and x2 into the single Xs list
num_points = len(x1)
Xs = num_points*[None]     ## training set
for i in range(num_points):
    Xs[i]=[x1_t[i],x2_t[i]]
    
## Test set with C=1 sigma^2=0.15
#
print "\n\nAttempting to generate LMs for test set using rbf kernel with C=1 sigma^2=0.15"
passed,missed,fn,fp = testClassifier(Xs,Ts,C1,Ls1,b1,K=rbf2)
accuracy=((float(num_points)-missed)/float(num_points))*100.0
if passed:
    print "  Check PASSED"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)

    
## Test set with C=1e6 sigma^2=0.15
#
print "\n\nAttempting to generate LMs for test set using rbf kernel with C=1e6 sigma^2=0.15"
passed,missed,fn,fp = testClassifier(Xs,Ts,C2,Ls2,b2,K=rbf2)
accuracy=((float(num_points)-missed)/float(num_points))*100.0
if passed:
    print "  Check PASSED"
print "Total of %d misclassification(s)." %(missed)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(accuracy)

##--------------------------------------------------------------------------
##
##  Plot data with expected classifications
##
##--------------------------------------------------------------------------
##

########################################################
##
## To implement figure out what zz is for this problem
##
## x = np.arange(-5, 5, 0.5)
## y = np.arange(-5, 5, 0.5)
## xx, yy = meshgrid(x, y, sparse=True)
## z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
## 
def plotContour(Xs, Ts, C, Ls, b):
    # prepare the x,y coords
    x = np.arange(-1,3.5,0.001)
    y = np.arange(-1.5,2.5,0.001)    
    xx,yy = np.meshgrid(x,y)
        
    
    rows = len(y)
    columns = len(x)
    # fill up z which classifies each xx and yy
    z = np.ndarray(shape=(rows,columns)) 
    
    
    for i in range(columns):
        for j in range(rows):
            z[j,i] = classify(array([x[i],y[j]]).tolist(), Xs, Ts, C, Ls, b, verbose=False)
  
    plt.contour(xx,yy,z)
    return xx,yy,z
   
## Organise training data to classifications
pos_ve = []     ## stores class 1 data
neg_ve = []     ## stores class 2 data
for i in range(num_points):
    if Ts[i] == 1:
        pos_ve.append([x1[i],x2[i]])
    else:
        neg_ve.append([x1[i],x2[i]])
        
plt.figure()
plt.scatter(*zip(*pos_ve), color='red')
plt.scatter(*zip(*neg_ve), color='blue')
X1,Y1,Z1=plotContour(Xs, Ts, C1, Ls1, b1)
savefig('Training Set C1')

## Organise testing data to classifications
pos_ve_t = []     ## stores class 1 data
neg_ve_t = []     ## stores class 2 data
for i in range(num_points):
    if Ts[i] == 1:
        pos_ve_t.append([x1_t[i],x2_t[i]])
    else:
        neg_ve_t.append([x1_t[i],x2_t[i]])
plt.figure()
plt.scatter(*zip(*pos_ve_t), color='red')
plt.scatter(*zip(*neg_ve_t), color='blue')
plotContour(Xs, Ts, C1, Ls1, b1)
savefig('Testing set C1')