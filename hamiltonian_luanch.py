from bit3.hamiltonian import fit_hamiltonian_parameters
import bit3.optimize as bit3_optimize
from bit3.write import save_fit_result_to_file
import queue
import threading
import numpy as np

# 脉冲条件设置
pulse_width = 30e-6  # 输入90°脉宽
P_exp1 = np.array([-1280, -1225, -1149, -1094, -60, 8, 74, 140, 822, 873, 893, 946])-8  # 三角座mini
# P_exp1 = np.array([-1345, -1294, -1215, -1164, -65, -2, 63,  132,  859,  910,  928, 979])+2  # 三角座pro
# 输入测得的热平衡谱各个峰位置，注意把中间第二个峰位置设置为0Hz！ 
o1 = 33.832900e6  # 三角座mini
# o1=35.450629e6  # 三角座pro
# 输入该机器F通道的共振频率
Type = 3  # 选择机型：1为三角座，2为三角座pro，3为三角座mini
calculatetype = 1  # 选择优化类型：1为计算新的脉冲文件，2为以旧的脉冲文件为初始值进行计算

# 创建队列
params_queue = queue.Queue()  
fit_result_queue = queue.Queue()  

# 启动参数拟合线程
fit_thread = threading.Thread(
    target=fit_hamiltonian_parameters,
    args=(params_queue, fit_result_queue, P_exp1, o1, Type)
)
fit_thread.start()
# 等待参数拟合完成
print("正在进行哈密顿量参数拟合...")
fit_thread.join()
# 获取拟合结果
fit_params = fit_result_queue.get()
if fit_params is not None:
    # 保存结果到fit_params.json（可修改filename为"fit_params"，无后缀）
    save_fit_result_to_file(fit_params, filename="fit_params.json")
else:
    print("拟合失败：线程返回空结果")