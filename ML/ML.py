import numpy as np
import random
import math
from PIL import Image

X = [] # list of colums
C = [] # list of
L = [] # list of matraces
B = [] # list of colums

def mat(A):
    return np.mat(A)

def r(i,j):
     return random.randint(i,j)


a = -100
b = 100
L1 = [[r(a,b) for j in range(100)] for i in range(10)]
L2 = [[r(a,b) for j in range(10)] for i in range(10)]
L3 = [[r(a,b) for j in range(10)] for i in range(2)]

L = [mat(L1),mat(L2),mat(L3)]

for l in L:
    B.append(mat([[r(a,b)]*1 for i in range(l.shape[0])]))

f = open("machine","r")


for i in range(len(L)):
    shape = L[i].shape
    for a in range(shape[0]):
        for b in range(shape[1]):
            s = f.readline()
            L[i][a,b] = float(s[0:len(s)-1])

for i in range(len(L)):
    shape = B[i].shape
    for j in range(shape[0]):
        s = f.readline()
        B[i][j,0] = float(s[0:len(s)-1])


answer = [[[0],[1]],[[1],[0]],[[0],[1]],[[1],[0]],[[0],[1]],[[1],[0]],[[0],[1]],[[1],[0]],[[0],[1]],[[1],[0]]]



for i in range(10):
    s = str(i) + ".jpg"
    img = Image.open(s)
    X.append([])
    C.append(answer[i])
    for j in range(10):
        for k in range(10):
            pix = img.getpixel((j,k))
            gray = (pix[0]+pix[1]+pix[2])/3
            X[i].append([gray])
    X[i] = mat(X[i])








def train(step):
    for i in range(len(L)):
        shape = L[i].shape
        for a in range(shape[0]):
            for b in range(shape[1]):
                d = d_er_f_div_d_L(i,a,b)
                L[i][a,b] -= d*step
        for j in range(len(B[i])):
            d = d_er_f_div_d_B(i,j)
            B[i][j,0] -= d*step

k = 100

def img_to_x(filename):
    img = Image.open(filename)
    Input = []
    for j in range(10):
        for k in range(10):
            pix = img.getpixel((j,k))
            gray = (pix[0]+pix[1]+pix[2])/3
            Input.append([gray])
    return mat(Input)

def check_log(filename):
    return f(img_to_x(filename))

def check(filename):
    temp = f(img_to_x(filename))
    if temp[0]>temp[1]:
        return "true"
    return "false"
    

def exp(x):
    if x>100:
        return math.exp(100)
    if x<-100:
        return math.exp(-100)
    return math.exp(x)

def sig(x):
    return 1/(1+exp(-x/k))

def Sig(X):
    return For(sig,X)

def d_sig(x):
    a = sig(s)
    return a*a*exp(-x)/k

def d_Sig(X):
    return For(d_sig,X)

def i_sig(y):
    return -math.ln(1/y-1)*k

def i_Sig(X):
    return For(i_sig,X)

def For(f,X):
    shape = X.shape
    return np.mat([[f(X[i,j]) for j in range(shape[1])] for i in range(shape[0])])

def id(x):
    return x

def copy(x):
    return For(id,x)

def f(x):
    y = copy(x)
    temp = range(len(L))
    for i in temp:
        y = Sig(L[i]*y+B[i])
    return y

def norm(x):
    sum = 0
    shape = x.shape
    for i in range(shape[0]):
        sum += x[i,0]*x[i,0]
    return math.sqrt(sum)

def er_f(L,B):
    sum = 0
    temp = len(X)
    for i in range(temp):
        sum += norm(f(X[i])-C[i])
    return sum / temp       

Del = 100

def d_er_f_div_d_L(i,a,b):
    er_f_0 = er_f(L,B)
    l = L[i][a,b]
    L[i][a,b] += Del
    er_f_1 = er_f(L,B)
    L[i][a,b] = l
    return (er_f_1-er_f_0) / Del

def d_er_f_div_d_B(i,j):
    er_f_0 = er_f(L,B)
    b = B[i][j,0]
    B[i][j,0] += Del
    er_f_1 = er_f(L,B)
    B[i][j,0] = b
    return (er_f_1-er_f_0) / Del


