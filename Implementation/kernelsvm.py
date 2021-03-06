import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
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



def makeB(Xs,Ts,C=1,Ls=None,K=rbf2):
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
                if Ls[i] >= 1e-10 and Ls[i] < C:
                    b_sum -= Ls[i] * Ts[i] * K(Xs[i],Xs[n])
            
    return b_sum/sv_count


def classify(x,Xs,Ts,C=1,Ls=None,b=None,K=rbf2,verbose=True,activation_get=False):
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
    
    if activation_get==True:
        return y
    else:
        if y > 0.0: return +1
        elif y < 0.0: return -1
        else: return 0 


def testClassifier(Xs,Ts,Xs_train=None,Ts_train=None,C=1,Ls=None,b=None,K=rbfKernel,verbose=True):
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
    ## Do classification test
    missed = 0
    fn = 0
    fp = 0
    
    # if SVM is already trained then use those values to classify
    if Xs_train != None and Ts_train != None:
        for i in range(len(Xs)):       
            c = classify(Xs[i],Xs_train,Ts_train,C,Ls,b,K=K)    
            if c != Ts[i]:
                missed += 1
                if c == 1 and Ts[i] == -1:
                    fp += 1
                elif c == -1 and Ts[i] == 1:
                    fn += 1
                if verbose:
                    print "Misclassification: input %s, output %d, expected %d" %\
                          (Xs[i],c,Ts[i])
    else:
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
    return missed, fn, fp

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
##  Result data storage
##
##--------------------------------------------------------------------------
##
missed=[]
false_neg=[]
false_pos=[]
accuracy=[]

def storeResults(miss,fn,fp,acc):
    missed.append(miss)    
    false_neg.append(fn)
    false_pos.append(fp)
    accuracy.append(acc)

def reportResults():
    print "\nREPORTING OVERALL PERFORMANCE"    
    c=[1,1e6]
    for i in range(2):
        print '   Training Set C={0}'.format(c[i])
        print """   Missed={0}. Accuracy={1}%. False Negative(s)={2}. False Positive(s)={3}
        """.format(missed[i],accuracy[i], false_neg[i], false_pos[i])
    for i in range(2):
        print '   Testing Set C={0}'.format(c[i])
        print """   Missed={0}. Accuracy={1}%. False Negative(s)={2}. False Positive(s)={3}
        """.format(missed[i+2],accuracy[i+2], false_neg[i+2], false_pos[i+2])
            
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
    miss,fn,fp = testClassifier(Xs,Ts,C=C1,Ls=Ls1,b=b1,K=rbf2)
    acc=((float(num_points)-miss)/float(num_points))*100.0
    
    #store results
    storeResults(miss,fn,fp,acc)
    
print "Total of %d misclassification(s)." %(miss)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(acc)
    
# Training set with C=1e6 sigma^2=0.15

C2 = 1e6
print "\n\nAttempting to generate LMs for training test using rbf kernel with C=1e6 sigma^2=0.15"
status,Ls2=makeLambdas(Xs,Ts,C2,K=rbf2)
if status == 'optimal':
    b2=makeB(Xs,Ts,C2,Ls2,K=rbf2)
    miss,fn,fp = testClassifier(Xs,Ts,C=C2,Ls=Ls2,b=b2,K=rbf2)
    acc=((float(num_points)-miss)/float(num_points))*100.0

    #store results   
    storeResults(miss,fn,fp,acc)
    
print "Total of %d misclassification(s)." %(miss)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(acc)

##--------------------------------------------------------------------------
##
##  Working with the testing set
##  Reuse the corresponding L's and C's from the training phase
##
##--------------------------------------------------------------------------
##
x1_t = []     ## stores x
x2_t = []     ## stores y
Ts_t = []     ## desired output of training set
with open('testing-dataset-aut-2016.txt') as f:
    for line in f:
        data = line.split()
        x1_t.append(float(data[0]))
        x2_t.append(float(data[1]))
        Ts_t.append(int(data[2]))
        

## Place x1 and x2 into the single Xs list
num_points = len(x1)
Xs_t = num_points*[None]     ## training set
for i in range(num_points):
    Xs_t[i]=[x1_t[i],x2_t[i]]
    
## Test set with C=1 sigma^2=0.15
#
print "\n\nAttempting to generate LMs for test set using rbf kernel with C=1 sigma^2=0.15"
miss,fn,fp = testClassifier(Xs_t,Ts_t,Xs,Ts,C1,Ls1,b1,K=rbf2)
acc=((float(num_points)-miss)/float(num_points))*100.0

# store results   
storeResults(miss,fn,fp,acc)

# report
print "Total of %d misclassification(s)." %(miss)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(acc)

    
## Test set with C=1e6 sigma^2=0.15
#
print "\n\nAttempting to generate LMs for test set using rbf kernel with C=1e6 sigma^2=0.15"
miss,fn,fp = testClassifier(Xs_t,Ts_t,Xs,Ts,C2,Ls2,b2,K=rbf2)
acc=((float(num_points)-miss)/float(num_points))*100.0

# store results   
storeResults(miss,fn,fp,acc)

print "Total of %d misclassification(s)." %(miss)
print "False Positive(s): %d, False Negative(s): %d" %(fn, fp) 
print "Overall accuracy %.2f%%" %(acc)

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
def plotContour(Xs, Ts, C, Ls, b, pos_ve, neg_ve, title, filled=False):
   # plt.figure(figsize=(6,5))    
    plt.figure()
    
    ## sort out misclassifications to correct data
    miss_pos_ve=[]
    corr_pos_ve=[]
    miss_neg_ve=[]
    corr_neg_ve=[]
    for i in range(len(pos_ve)):
        if classify(pos_ve[i], Xs, Ts, C, Ls, b, verbose=False) == -1:
            miss_pos_ve.append(pos_ve[i])
        else:
            corr_pos_ve.append(pos_ve[i])
        if classify(neg_ve[i], Xs, Ts, C, Ls, b, verbose=False) == 1:
            miss_neg_ve.append(neg_ve[i])
        else:
            corr_neg_ve.append(neg_ve[i])
    
    ## Contour section
    # prepare the x,y coords
    step=0.01
    x = np.arange(-1.5,4,step)
    y = np.arange(-1.9,3.5,step)  
    xx,yy = np.meshgrid(x,y)
            
    rows = len(y)
    columns = len(x)
    
    # fill up z which classifies each xx and yy
    z = np.ndarray(shape=(rows,columns)) 
    
    # store activation levels in the 2-D Z array
    for i in range(columns):
        for j in range(rows):
            z[j,i] = classify(([x[i],y[j]]), Xs, Ts, C, Ls, b, verbose=False, activation_get=True)
    
    
    if filled == False:
        plt.contour(xx,yy,z,levels=[-1,0,1], colors=['b','g','r'])
    else:
        plt.contour(xx,yy,z,levels=[-1,0,1], colors=['b','g','r'])
        cmap = clr.ListedColormap(['b','r'])
        plt.contourf(xx,yy,z,0, cmap=cmap,alpha=0.2)
        
    ## Points section
    marker_size=10
    line_width=.5
    plt.scatter(*zip(*corr_pos_ve),s=marker_size, color='r', edgecolor='black', linewidth=line_width)
    plt.scatter(*zip(*corr_neg_ve),s=marker_size, color='b',edgecolor='black', linewidth=line_width)
    if len(miss_pos_ve) != 0:
        plt.scatter(*zip(*miss_pos_ve),s=marker_size,color='r',  marker='D',edgecolor='black', linewidth=line_width)
    if len(miss_pos_ve) != 0:    
        plt.scatter(*zip(*miss_neg_ve),s=marker_size,color='b', marker='D',edgecolor='black', linewidth=line_width)
    plt.title(title)
    
        
    #savefig(title+'.png',dpi=1000)    

## Organise training data to classifications
pos_ve = []     ## stores class 1 data
neg_ve = []     ## stores class 2 data
for i in range(num_points):
    if Ts[i] == 1:
        pos_ve.append([x1[i],x2[i]])
    else:
        neg_ve.append([x1[i],x2[i]])
        
## training set contour plots
plotContour(Xs, Ts, C1, Ls1, b1, pos_ve, neg_ve,title="Training Set C=1, Sigma$^2$=0.15")
plotContour(Xs, Ts, C2, Ls2, b2, pos_ve, neg_ve, title="Training Set C=1e6, Sigma^2=0.15")

## trainng set filled contour plots
plotContour(Xs, Ts, C1, Ls1, b1, pos_ve, neg_ve, title="Training Set C=1, Sigma$^2$=0.15 (Filled)",filled=True)
plotContour(Xs, Ts, C2, Ls2, b2, pos_ve, neg_ve, title="Training Set C=1e6, Sigma$^2$=0.15 (Filled)",filled=True)

## Organise testing data to classifications
pos_ve_t = []     ## stores class 1 data
neg_ve_t = []     ## stores class 2 data
for i in range(num_points):
    if Ts_t[i] == 1:
        pos_ve_t.append([x1_t[i],x2_t[i]])
    else:
        neg_ve_t.append([x1_t[i],x2_t[i]])

## testing set contour plots
plotContour(Xs, Ts, C1, Ls1, b1, pos_ve_t, neg_ve_t, title="Testing Set C=1, Sigma$^2$=0.15")
plotContour(Xs, Ts, C2, Ls2, b2, pos_ve_t, neg_ve_t, title="Testing Set C=1e6, Sigma$^2$=0.15")

## testing set filled contour plots
plotContour(Xs, Ts, C1, Ls1, b1, pos_ve_t, neg_ve_t, title="Testing Set C=1, Sigma$^2$=0.15 (Filled)",filled=True)
plotContour(Xs, Ts, C2, Ls2, b2, pos_ve_t, neg_ve_t, title="Testing Set C=1e6, Sigma$^2$=0.15 (Filled)",filled=True)

reportResults()