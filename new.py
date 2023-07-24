# 导入相关的库和模块
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定义相关的参数
L = 1000 # 道路长度
W = 10 # 道路宽度
dx = dy = 1 # 网格大小
dt = 1 # 时间步长
NA = 10 # 应急车辆数量
NB = 100 # 社会车辆数量
vA = 20 # 应急车辆速度
vB = 10 # 社会车辆速度
aA = 2 # 应急车辆加速度
aB = 1 # 社会车辆加速度
dA = 20 # 应急车辆刹车距离
dB = 10 # 社会车辆刹车距离
pA = 0.8 # 应急车辆换道概率
pB = 0.5 # 社会车辆换道概率
sigmaA = sigmaB = 1 # 概率因素的标准差
tauA = 0.5 # 松弛时间参数A
tauB = 0.8 # 松弛时间参数B
gA = gB = [0, -1] # 外部重力项
fA = fB = [0, -1] # 外部摩擦项

# 定义九个方向上的单位向量和权重系数
e = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
w = np.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])

# 定义平衡态下的分布函数
def f_eq(rho, u):
    # f_eq = np.zeros((9,W,L))
    f_eq = np.ones((9,W,L))
    # for i in range(9):
    #     # f_eq[i,:,:] = rho * w[i] * (1 + 3 * e[i].dot(u) + 4.5 * (e[i].dot(u))**2 - 1.5 * u.dot(u))
    #     f_eq[i,:,:] = np.expand_dims(f_eq[i,:,:], axis=0) * np.expand_dims(rho, axis=0) * np.expand_dims(w[i], axis=(0,1)) * (1 + 3 * e[i].dot(u) + 4.5 * (e[i].dot(u))**2 - 1.5 * u.dot(u))
    # return f_eq

    for i in range(9): # 遍历九个方向上的索引i 
        for j in range(W): # 遍历网格长度上的索引j 
            for k in range(L): # 遍历网格宽度上的索引k 
                f_eq[i,j,k] = rho[j,k] * w[i] * (1 + 3 * e[i].dot(u[:,j,k]) + 4.5 * (e[i].dot(u[:,j,k]))**2 - 1.5 * u[:,j,k].dot(u[:,j,k])) # 根据公式计算f_eq[i,j,k]的值
    return f_eq

# 定义外部力项和反馈项
def F(rho, g, f,u):
    # F = np.zeros((9,W,L))
    F = np.ones((9,W,L))
    # for i in range(9):
    #     F[i,:,:] = w[i] * (rho * g + rho * f) * (1 + 3 * e[i].dot(u))
        # F[i,:,:] = w[i] * np.transpose(np.sum(rho[:,:,np.newaxis] * g +rho[:,np.newaxis,np.newaxis] * f,axis=1),(1,0,2)) * (1 + 3 * np.tensordot(e[i],u,axes=([0],[0])))
    
    # for i in range(9): # 遍历九个方向上的索引i 
    #     for j in range(W): # 遍历网格长度上的索引j 
    #         for k in range(L): # 遍历网格宽度上的索引k 
    #             # F[i,j,k] = w[i] * (rho[j,k] * g + rho[j,k] * f[:,j,k]) * (1 + 3 * e[i].dot(u[:,j,k]))
    #             F[i,j,k] = w[i] * (np.array(rho[j,k],dtype=np.float64).dot(g) + np.array(rho[j,k],dtype=np.float64) * f[i,j,k]) * (1 + 3 * np.tensordot(e[i],u[:,j,k],axes=([0],[0])))
 
    return F

def R(rho, g, f, u_other):
    # R = np.zeros((9,W,L))
    R = np.ones((9,W,L))
    # for i in range(9):
    #     R[i,:,:] = -w[i] * (rho * g + rho * f) * (1 + 3 * e[i].dot(u_other))
    return R

# 定义碰撞步骤
def collision(fA, fB):
    rhoA = np.sum(fA,axis=0)
    rhoB = np.sum(fB,axis=0)
    uA = np.tensordot(e,fA,axes=([0],[0])) / rhoA
    uB = np.tensordot(e,fB,axes=([0],[0])) / rhoB
    
    fAeq = f_eq(rhoA,uA)
    fBeq = f_eq(rhoB,uB)
    
    FA = F(rhoA,gA,fA,uA)
    FB = F(rhoB,gB,fB,uB)
    
    RA = R(rhoA,gA,fA,uB)
    RB = R(rhoB,gB,fB,uA)
    
    fAs = fA - (fA - fAeq) / tauA + FA + RA
    fBs = fB - (fB - fBeq) / tauB + FB + RB
    
    return fAs, fBs

# 定义传播
def propagation(fA, fB):
    fAn = np.zeros((9,W,L))
    fBn = np.zeros((9,W,L))
    for i in range(9):
        fAn[i,:,:] = np.roll(fA[i,:,:],e[i],axis=(0,1))
        fBn[i,:,:] = np.roll(fB[i,:,:],e[i],axis=(0,1))
    return fAn, fBn

# 定义概率因素和改变后的方向
def probability(u, v, sigma):
    P = np.zeros((9,W,L))
    # for i in range(9):
    #     P[i,:,:] = np.exp(-(e[i].dot(u) - v)**2 / (2 * sigma**2))
    return P

def change_direction(f, u, v, p):
    fc = np.zeros((9,W,L))
    # for i in range(9):
    #     ip = np.argmin(np.abs(e.dot(u) - v),axis=0)
    #     fc[i,:,:] = (1 - p[i,:,:]) * f[i,:,:] + p[i,:,:] * f[ip,:,:]
    return fc

# 定义边界条件
def boundary(fA, fB):
    # 周期性边界条件
    fA[:,0,:] = fA[:,-1,:]
    fA[:,-1,:] = fA[:,0,:]
    fB[:,0,:] = fB[:,-1,:]
    fB[:,-1,:] = fB[:,0,:]
    
    # 反弹边界条件
    fA[2,0,:] = fA[4,1,:]
    fA[4,0,:] = fA[2,1,:]
    fA[5,0,:] = fA[7,1,:]
    fA[6,0,:] = fA[8,1,:]
    fA[7,0,:] = fA[5,1,:]
    fA[8,0,:] = fA[6,1,:]
    
    fA[2,-1,:] = fA[4,-2,:]
    fA[4,-1,:] = fA[2,-2,:]
    fA[5,-1,:] = fA[7,-2,:]
    fA[6,-1,:] = fA[8,-2,:]
    fA[7,-1,:] = fA[5,-2,:]
    fA[8,-1,:] = fA[6,-2,:]
    
    fB[2,0,:] = fB[4,1,:]
    fB[4,0,:] = fB[2,1,:]
    fB[5,0,:] = fB[7,1,:]
    fB[6,0,:] = fB[8,1,:]
    fB[7,0,:] = fB[5,1,:]
    fB[8,0,:] = fB[6,1,:]
    
    fB[2,-1,:] = fB[4,-2,:]
    fB[4,-1,:] = fB[2,-2,:]
    fB[5,-1,:] = fB[7,-2,:]
    fB[6,-1,:] = fB[8,-2,:]
    fB[7,-1,:] = fB[5,-2,:]
    fB[8,-1,:] = fB[6,-2,:]
    
    return fA, fB

# 定义初始化函数
def initialize():
    global uA,uB
    # 随机生成初始分布函数
    fA = np.random.rand(9,W,L)
    fB = np.random.rand(9,W,L)
    # 根据初始分布函数计算初始密度和速度
    rhoA = np.sum(fA,axis=0)
    rhoB = np.sum(fB,axis=0)
    uA = np.tensordot(e,fA,axes=([0],[0])) / rhoA
    uB = np.tensordot(e,fB,axes=([0],[0])) / rhoB
    # 根据初始密度和速度计算初始平衡态分布函数
    fAeq = f_eq(rhoA,uA)
    fBeq = f_eq(rhoB,uB)
    # 使初始分布函数接近平衡态分布函数
    fAs,fBs=collision(fA,fB)
    return fAs, fBs

# 定义主循环函数
def main_loop(T):
    global uA,uB
    # 初始化分布函数
    fAs, fBs = initialize()
    # 创建空列表存储结果
    rhoAs = []
    rhoBs = []
    uAs = []
    uBs = []
    # 迭代T次
    for t in range(T):
        # 碰撞步骤
        fAs, fBs = collision(fAs, fBs)
        # 传播步骤
        fAn, fBn = propagation(fAs, fBs)
        # 概率因素和改变后的方向
        PA = probability(uA, vA, sigmaA)
        PB = probability(uB, vB, sigmaB)
        fAc = change_direction(fAn, uA, vA, PA)
        fBc = change_direction(fBn, uB, vB, PB)
        # 边界条件
        fAs, fBs = boundary(fAc, fBc)
        # 计算密度和速度
        rhoA = np.sum(fAs,axis=0)
        rhoB = np.sum(fBs,axis=0)
        uA = np.tensordot(e,fAs,axes=([0],[0])) / rhoA
        uB = np.tensordot(e,fBs,axes=([0],[0])) / rhoB
        # 存储结果
        rhoAs.append(rhoA)
        rhoBs.append(rhoB)
        uAs.append(uA)
        uBs.append(uB)
    return rhoAs, rhoBs, uAs, uBs

def plot_results(rhoAs, rhoBs, uAs, uBs):
    # 将结果转换为数据框格式
    df_rhoA = pd.DataFrame(np.array(rhoAs).reshape(-1,W*L))
    df_rhoB = pd.DataFrame(np.array(rhoBs).reshape(-1,W*L))
    df_uA = pd.DataFrame(np.array(uAs).reshape(-1,W*L))
    df_uB = pd.DataFrame(np.array(uBs).reshape(-1,W*L))
    
    # 绘制密度随时间的变化曲线图
    plt.figure(figsize=(10,5))
    plt.plot(df_rhoA.mean(axis=1),label='rho_A')
    plt.plot(df_rhoB.mean(axis=1),label='rho_B')
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.title('Density vs Time')
    plt.legend()
    
    # 绘制速度随时间的变化曲线图
    plt.figure(figsize=(10,5))
    plt.plot(df_uA.mean(axis=1),label='u_A')
    plt.plot(df_uB.mean(axis=1),label='u_B')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Velocity vs Time')
    plt.legend()
    
    # 绘制密度随空间的变化热力图
    plt.figure(figsize=(10,5))
    plt.imshow(df_rhoA.iloc[-1,:].values.reshape(W,L),cmap='jet',aspect='auto')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Density vs Space (t=T)')
     
     # 绘制速度随空间的变化热力图
    plt.figure(figsize=(10,5))
    plt.imshow(df_uA.iloc[-1,:].values.reshape(W,L),cmap='jet',aspect='auto')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity vs Space (t=T)')

    plt.tight_layout()
    plt.show()

    return 0

rhoAs, rhoBs, uAs, uBs = main_loop(1000)
plot_results(rhoAs, rhoBs, uAs, uBs)
 
