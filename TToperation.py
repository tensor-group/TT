import torch
import numpy as np

def Hadamared(matrixA, matrixB, returnTorch=False):
    assert len(matrixA) == len(matrixB)
    for index in range(len(matrixA)):
        assert matrixA.shape[index] == matrixB.shape[index]
    if isinstance(matrixA, torch.Tensor):
        matrixA = matrixA.numpy()
    if isinstance(matrixB, torch.Tensor):
        matrixB = matrixB.numpy()
    if returnTorch:
        return torch.from_numpy(np.multiply(matrixA, matrixB))
    return np.multiply(matrixA, matrixB)

def TTAdd(TTListA, TTListB):
    assert len(TTListA) == len(TTListB)
    TTList_result = []
    for i in range(len(TTListA)):
        TTcoreA = TTListA[i]
        TTcoreB = TTListB[i]
        if i == 0:
            TTList_result.append(np.concatenate((TTcoreA, TTcoreB)))
        elif i == len(TTListA)-1:
            assert TTcoreA.shape[1] == TTcoreB.shape[1]
            TTList_result.append(np.concatenate((np.concatenate((TTcoreA, np.empty_like(TTcoreA))), np.concatenate((np.empty_like(TTcoreB), TTcoreB))), axis=1))
        else:
            TTList_result.append(np.concatenate((TTcoreA, TTcoreB)).transpose())
    return TTList_result

def TTScalar(TTList, scalar, dim):
    TTList[dim] = TTList[dim] * scalar
    return TTList

def TTContraction(TTList, VectorList):
    pass

def TTMultiContraction(TTList, VectorList):
    assert len(TTList) == len(VectorList)
    W = []
    for i in range(len(TTList)):
        Temp = []
        for j in range(i, len(TTList)+1):
            Temp.append(()) #ui(ik)Ak(ik)是平常的矩阵积？但是Ak是一个三维数组啊？
    pass

def TTHadamard(TTListA, TTListB):
    assert len(TTListA) == len(TTListB)
    TTList_result = []
    for index in range(len(TTListA)):
        assert TTListA[index].shape == TTListB[index].shape
        TTList_result.append(Kronecker(TTListA[index], TTListB[index]))
    return TTList_result

def TTInner(TTListA, TTListB):
    assert len(TTListA) == len(TTListB)
    W = []
    for i in range(len(TTListA)):
        Temp = []
        for j in range(i):
            Temp.append(Kronecker(TTListA[j], TTListB[j], returnTorch=False))
        Tk = sum(np.array(Temp))
        W.append(Tk)
    return W

def TTNorm(TTList):
    pass

def TTDot(TTListA, TTListB):
    #Hadamard product
    pass

def TTMatrixbyVector(TTList, Vectorlist):
    pass

def Kronecker(tensorA, tensorB, returnTorch=False):
    assert len(tensorA.shape) == len(tensorB.shape) #保证维度的数量相同，比如都是二维矩阵或者都是三阶张量
    if isinstance(tensorA, torch.Tensor):
        tensorA = tensorA.numpy()
    if isinstance(tensorB, torch.Tensor):
        tensorB = tensorB.numpy()
    tensorC = np.kron(tensorA, tensorB)
    for index, Cshape in enumerate(tensorC.shape):
        assert Cshape == tensorA.shape[index] * tensorB.shape[index] #检查对应下标为index的结果张量的维度是否是两个乘子张量对应index的下标的维度的积
    if returnTorch:
        return torch.from_numpy(tensorC)
    return tensorC

a1 = np.random.random((3,3))
a2 = np.random.random((3,2,4))
b1 = np.random.random((2,4))
b2 = np.random.random((2,4,1))
ttcorea = [a2, a1]
ttcoreb = [b2, b1]