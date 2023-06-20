# 导入相关库和模块
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 定义晶格流体模型中的局部平衡分布函数
def f_ij_eq(x,y,v,a,vtype,i,t):
    rho_i = x[i,t] / L # 计算沿着纵向位置的流体密度
    u_i = v[i,t] / v_max # 计算沿着纵向位置的流体速度
    w_i = y[i,t] / W # 计算沿着横向位置的流体密度
    alpha_i = np.ones(9) # 初始化偏好因子向量
    alpha_i[0] = 0.5 # 设置静止方向的偏好因子为0.5
    alpha_i[2] = alpha_i[4] = alpha_i[6] = alpha_i[8] = 0.75 # 设置对角线方向的偏好因子为0.75
    if vtype[i] == 0: # 如果是社会车辆
        alpha_i[3] = alpha_i[7] = 0.25 # 设置横向方向的偏好因子为0.25
        alpha_i[1] = alpha_i[5] = 0.75 # 设置纵向方向的偏好因子为0.75
    else: # 如果是应急车辆
        alpha_i[3] = alpha_i[7] = 0.75 # 设置横向方向的偏好因子为0.75
        alpha_i[1] = alpha_i[5] = 1.25 # 设置纵向方向的偏好因子为1.25
    c_s = np.sqrt(3) / 2 # 设置声速，取D2Q9模型中的值
    f_ij_eq = np.zeros(9) # 初始化局部平衡分布函数向量
    for j in range(9): # 遍历九个可能的速度方向
        e_jx = np.cos(np.pi / 4 * j) # 计算沿着纵向位置的速度分量
        e_jy = np.sin(np.pi / 4 * j) # 计算沿着横向位置的速度分量
        f_ij_eq[j] = rho_i * w_i * (1 + (e_jx * u_i + e_jy * w_i) / c_s ** 2 + ((e_jx * u_i + e_jy * w_i) ** 2 - (u_i ** 2 + w_i ** 2)) / (2 * c_s ** 4)) * alpha_i[j] # 计算局部平衡分布函数
    return f_ij_eq

# 定义晶格流体模型中的松弛时间参数
def tau_ij(x,y,v,a,vtype,i,t):
    omega_i = np.ones(9) # 初始化碰撞参数向量
    omega_i[0] = 0.5 # 设置静止方向的碰撞参数为0.5
    omega_i[2] = omega_i[4] = omega_i[6] = omega_i[8] = 0.75 # 设置对角线方向的碰撞参数为0.75
    if vtype[i] == 0: # 如果是社会车辆
        omega_i[3] = omega_i[7] = 1 # 设置横向方向的碰撞参数为1
        omega_i[1] = omega_i[5] = 0.75 # 设置纵向方向的碰撞参数为0.75
    else: # 如果是应急车辆
        omega_i[3] = omega_i[7] = 0.5 # 设置横向方向的碰撞参数为0.5
        omega_i[1] = omega_i[5] = 0.25 # 设置纵向方向的碰撞参数为0.25
    tau_ij = 1 / omega_i # 计算松弛时间参数
    return tau_ij

# 定义晶格流体模型中的流体密度分布函数
def f_ij(x,y,v,a,vtype,i,t):
    f_ij_eq = f_ij_eq(x,y,v,a,vtype,i,t) # 计算局部平衡分布函数
    tau_ij = tau_ij(x,y,v,a,vtype,i,t) # 计算松弛时间参数
    f_ij = np.zeros(9) # 初始化流体密度分布函数向量
    for j in range(9): # 遍历九个可能的速度方向
        e_jx = np.cos(np.pi / 4 * j) # 计算沿着纵向位置的速度分量
        e_jy = np.sin(np.pi / 4 * j) # 计算沿着横向位置的速度分量
        x_j = x[i,t] - e_jx * v[i,t+1] # 计算沿着纵向位置的传播位置
        y_j = y[i,t] - e_jy * v[i,t+1] # 计算沿着横向位置的传播位置
        k_j = np.argmin(np.abs(x[:,t+1] - x_j)) # 找到最接近传播位置的车辆编号
        if np.abs(x[k_j,t+1] - x_j) < v_max: # 如果最接近传播位置的车辆距离小于最大速度（即有可能发生碰撞）
            f_ij[k_j,j,t+1] = f_ij[i,j,t+1] - (f_ij[i,j,t+1] - f_ij[k_j,j,t+1]) / tau_ij[k_j,j,t+1] # 更新流体密度分布函数，考虑碰撞过程
        else: # 如果最接近传播位置的车辆距离大于最大速度（即没有发生碰撞）
            f_ij[k_j,j,t+1] = f_ij[i,j,t+1] - (f_ij[i,j,t+1] - f_ij_eq[k_j,j,t+1]) / tau_ij[k_j,j,t+1] # 更新流体密度分布函数，考虑传播过程
    return f_ij


# 生成模拟数据
np.random.seed(0) # 设置随机数种子，保证每次运行结果一致
N = 100 # 设置车辆数量
T = 100 # 设置时间步长数量
L = 1000 # 设置路段长度
W = 12 # 设置路段宽度
M = 3 # 设置车道数量
v_max = 20 # 设置最大速度
a_max = 2 # 设置最大加速度
p = 0.1 # 设置应急车辆占比
x = np.zeros((N,T)) # 初始化车辆位置矩阵
y = np.zeros((N,T)) # 初始化车辆横向位置矩阵
v = np.zeros((N,T)) # 初始化车辆速度矩阵
a = np.zeros((N,T)) # 初始化车辆加速度矩阵
u = np.zeros((N,T)) # 初始化车辆控制输入矩阵
t_e = np.zeros(N) # 初始化应急车辆行驶时间向量
t_s = np.zeros(N) # 初始化社会车辆行驶时间向量
vtype = np.random.binomial(1,p,N) # 随机生成车辆类型向量，0表示社会车辆，1表示应急车辆
v[:,0] = v_max * np.random.rand(N) # 随机生成初始速度向量
a[:,0] = a_max * (2 * np.random.rand(N) - 1) # 随机生成初始加速度向量
y[:,0] = W / M * (np.random.randint(M, size=N) + 0.5) # 随机生成初始横向位置向量

# 模拟车辆运动过程
for i in range(N):
    for t in range(T-1):
        if x[i,t] < L: # 如果车辆还没有到达路段终点
            
            if vtype[i] == 0: # 如果是社会车辆
                
                if np.random.rand() < 0.01: # 如果有一定概率换道
                    
                    lane_i = int(y[i,t] / (W / M)) # 获取当前所在的车道编号
                    
                    if lane_i == 0: # 如果在最左侧的车道
                        
                        lane_j = lane_i + 1 # 获取右侧的车道编号
                        
                        if np.sum((y[:,t] == W / M * (lane_j + 0.5)) & (x[:,t] > x[i,t])) == 0: # 如果右侧的车道没有前方的车辆
                        
                            y[i,t+1] = W / M * (lane_j + 0.5) # 换到右侧的车道
                        
                        else: # 如果右侧的车道有前方的车辆
                        
                            y[i,t+1] = y[i,t] # 保持在当前车道
                        
                    elif lane_i == M - 1: # 如果在最右侧的车道
                        
                        lane_j = lane_i - 1 # 获取左侧的车道编号
                        
                        if np.sum((y[:,t] == W / M * (lane_j + 0.5)) & (x[:,t] > x[i,t])) == 0: # 如果左侧的车道没有前方的车辆
                        
                            y[i,t+1] = W / M * (lane_j + 0.5) # 换到左侧的车道
                        
                        else: # 如果左侧的车道有前方的车辆
                        
                            y[i,t+1] = y[i,t] # 保持在当前车道
                        
                    else: # 如果在中间的车道
                        
                        lane_j = np.random.choice([lane_i - 1, lane_i + 1]) # 随机选择左侧或右侧的车道编号
                        
                        if np.sum((y[:,t] == W / M * (lane_j + 0.5)) & (x[:,t] > x[i,t])) == 0: # 如果选择的车道没有前方的车辆
                        
                            y[i,t+1] = W / M * (lane_j + 0.5) # 换到选择的车道
                        
                        else: # 如果选择的车道有前方的车辆
                        
                            y[i,t+1] = y[i,t] # 保持在当前车道
                
                else: # 如果没有换道
                
                    y[i,t+1] = y[i,t] # 保持在当前横向位置
            
            else: # 如果是应急车辆
                
                if np.random.rand() < 0.05: # 如果有一定概率换道
                    
                    lane_i = int(y[i,t] / (W / M)) # 获取当前所在的车道编号
                    
                    if lane_i == 0: # 如果在最左侧的车道
                        
                        lane_j = lane_i + 1 # 获取右侧的车道编号
                        
                        if np.sum((y[:,t] == W / M * (lane_j + 0.5)) & (x[:,t] > x[i,t])) == 0: # 如果右侧的车道没有前方的车辆
                        
                            y[i,t+1] = W / M * (lane_j + 0.5) # 换到右侧的车道
                        
                        else: # 如果右侧的车道有前方的车辆
                        
                            y[i,t+1] = y[i,t] # 保持在当前车道
                        
                    elif lane_i == M - 1: # 如果在最右侧的车道
                        
                        lane_j = lane_i - 1 # 获取左侧的车道编号
                        
                        if np.sum((y[:,t] == W / M * (lane_j + 0.5)) & (x[:,t] > x[i,t])) == 0: # 如果左侧的车道没有前方的车辆
                        
                            y[i,t+1] = W / M * (lane_j + 0.5) # 换到左侧的车道
                        
                        else: # 如果左侧的车道有前方的车辆
                        
                            y[i,t+1] = y[i,t] # 保持在当前车道
                        
                    else: # 如果在中间的车道
                        
                        lane_j = np.random.choice([lane_i - 1, lane_i + 1]) # 随机选择左侧或右侧的车道编号
                        
                        if np.sum((y[:,t] == W / M * (lane_j + 0.5)) & (x[:,t] > x[i,t])) == 0: # 如果选择的车道没有前方的车辆
                        
                            y[i,t+1] = W / M * (lane_j + 0.5) # 换到选择的车道
                        
                        else: # 如果选择的车道有前方的车辆
                        
                            y[i,t+1] = y[i,t] # 保持在当前车道
                
                else: # 如果没有换道
                
                    y[i,t+1] = y[i,t] # 保持在当前横向位置
            
            u[i,t] = a_max * (2 * np.random.rand() - 1) # 随机生成控制输入
            a[i,t+1] = a[i,t] + u[i,t] - (f_ij(x,y,v,a,vtype,i,t) - f_ij_eq(x,y,v,a,vtype,i,t)) / tau_ij(x,y,v,a,vtype,i,t) # 更新加速度，考虑晶格流体模型中的传播和碰撞过程
            v[i,t+1] = v[i,t] + a[i,t+1] # 更新速度
            x[i,t+1] = x[i,t] + v[i,t+1] # 更新位置

            if a[i,t+1] > a_max: # 如果加速度超过最大加速度
                a[i,t+1] = a_max # 限制加速度为最大加速度
            if a[i,t+1] < -a_max: # 如果加速度低于最小加速度（即最大减速度）
                a[i,t+1] = -a_max # 限制加速度为最小加速度（即最大减速度）
            if v[i,t+1] > v_max: # 如果速度超过最大速度
                v[i,t+1] = v_max # 限制速度为最大速度
            if v[i,t+1] < 0: # 如果速度低于零（即倒车）
                v[i,t+1] = 0 # 限制速度为零（即停止）
            if x[i,t+1] > L: # 如果位置超过路段长度（即到达终点）
                x[i,t+1] = L # 限制位置为路段长度（即终点）
                if vtype[i] == 0: # 如果是社会车辆
                    t_s[i] = t + 1 # 记录行驶时间
                else: # 如果是应急车辆
                    t_e[i] = t + 1 # 记录行驶时间



# 计算交通系统性能指标
flow_e = np.sum(vtype) / T / L * 3600 * 1000 # 计算应急车辆流量，单位为辆/小时/千米
flow_s = np.sum(1 - vtype) / T / L * 3600 * 1000 # 计算社会车辆流量，单位为辆/小时/千米
speed_e = np.mean(v[vtype == 1,:]) # 计算应急车辆平均速度，单位为米/秒
speed_s = np.mean(v[vtype == 0,:]) # 计算社会车辆平均速度，单位为米/秒
density_e = np.mean(x[vtype == 1,:]) / L # 计算应急车辆平均密度，单位为辆/米
density_s = np.mean(x[vtype == 0,:]) / L # 计算社会车辆平均密度，单位为辆/米
delay_s = np.mean(t_s[vtype == 0] - L / v_max) # 计算社会车辆平均延误，单位为秒
stop_s = np.sum(v[vtype == 0,:] == 0) / np.sum(1 - vtype) # 计算社会车辆平均停车次数，单位为次
time_e = np.mean(t_e[vtype == 1]) # 计算应急车辆平均行驶时间，单位为秒
time_s = np.mean(t_s[vtype == 0]) # 计算社会车辆平均行驶时间，单位为秒
time_tot = np.sum(t_e) + np.sum(t_s) # 计算所有车辆总行驶时间，单位为秒
occupancy = np.mean(np.sum(x < L, axis=0)) / N # 计算道路占有率，单位为百分比
safety = 1 - np.sum(np.diff(x, axis=0) < 0) / (N * (N - 1) / 2) # 计算安全性，单位为百分比

# 绘制交通系统数据和结果的图形
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(x.T)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Position of vehicles over time")
plt.subplot(2,2,2)
plt.plot(v.T)
plt.xlabel("Time")
plt.ylabel("Speed")
plt.title("Speed of vehicles over time")
plt.subplot(2,2,3)
plt.plot(a.T)
plt.xlabel("Time")
plt.ylabel("Acceleration")
plt.title("Acceleration of vehicles over time")
plt.subplot(2,2,4)
plt.plot(u.T)
plt.xlabel("Time")
plt.ylabel("Control input")
plt.title("Control input of vehicles over time")
plt.tight_layout()
plt.show()

# 输出交通系统性能指标的表格
data = {"Flow (veh/h/km)": [flow_e, flow_s],
        "Speed (m/s)": [speed_e, speed_s],
        "Density (veh/m)": [density_e, density_s],
        "Delay (s)": [np.nan, delay_s],
        "Stop (times)": [np.nan, stop_s],
        "Time (s)": [time_e, time_s]}
index = ["Emergency vehicle", "Social vehicle"]
df = pd.DataFrame(data, index=index)
print(df)

# 输出交通系统总行驶时间、道路占有率和安全性的值
print(f"Total travel time: {time_tot} s")
print(f"Road occupancy: {occupancy * 100} %")
print(f"Safety: {safety * 100} %")
