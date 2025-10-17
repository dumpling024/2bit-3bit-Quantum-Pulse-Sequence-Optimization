import numpy as np
from scipy.linalg import expm, kron, sqrtm
from scipy.optimize import minimize, LinearConstraint, Bounds
import threading
import queue
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings("ignore")

# 全局变量定义
class GlobalVars:
     def __init__(self):
        self.U_theo = []    
        self.I = np.array([[1, 0], [0, 1]], dtype=np.complex64)
        self.X = np.array([[0, 1], [1, 0]], dtype=np.complex64) 
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64) 
        self.Z = np.array([[1, 0], [0, -1]], dtype=np.complex64) 
        self.Ix = self.X / 2
        self.Iy = self.Y / 2
        self.Iz = self.Z / 2
        self.H_int = []
        self.H_pulseHx = []
        self.H_pulseHy = []
        self.H_pulsePx = []
        self.H_pulsePy = []
        self.state_theo = []
        self.rho0_6 = []
        self.rho0_3 = []
        self.rho0_4 = []
        self.rho0_PPS1_PostRelaxation = None
        self.rho0_PPS1_3 = None
        self.rho0_PPS1_8 = None
        self.pulsenumber = None
        self.optim_type = 2
        print("GlobalVars 初始化完成")
        self.t_Q1_X90 = np.array([])
        self.theta_Q1_X90 = np.array([])
        self.t_Q1_Y90 = np.array([])
        self.theta_Q1_Y90 = np.array([])
        self.t_Q1_X90N = np.array([])
        self.theta_Q1_X90N = np.array([])
        self.t_Q1_Y90N = np.array([])
        self.theta_Q1_Y90N = np.array([])

        self.t_Q2_X90 = np.array([])
        self.theta_Q2_X90 = np.array([])
        self.t_Q2_Y90 = np.array([])
        self.theta_Q2_Y90 = np.array([])
        self.t_Q2_X90N = np.array([])
        self.theta_Q2_X90N = np.array([])
        self.t_Q2_Y90N = np.array([])
        self.theta_Q2_Y90N = np.array([])

        self.t_Q1_X180 = np.array([])
        self.theta_Q1_X180 = np.array([])
        self.t_Q1_Y180 = np.array([])
        self.theta_Q1_Y180 = np.array([])

        self.t_Q2_X180 = np.array([])
        self.theta_Q2_X180 = np.array([])
        self.t_Q2_Y180 = np.array([])
        self.theta_Q2_Y180 = np.array([])

        self.t_Q1_H = np.array([])
        self.theta_Q1_H = np.array([])
        self.t_Q2_H = np.array([])
        self.theta_Q2_H = np.array([])

        self.t_CNOT12 = np.array([])
        self.theta_CNOT12 = np.array([])
        self.t_CNOT21 = np.array([])
        self.theta_CNOT21 = np.array([])
        self.t_CZ = np.array([])
        self.theta_CZ = np.array([])
        self.t_SWAP = np.array([])
        self.theta_SWAP = np.array([])
        self.t_PPS_PART1 = np.array([])
        self.theta_PPS_PART1 = np.array([])
        self.U_permute = np.array([])
        self.fval = 0

def roundn(x, decimals):
    """正确实现MATLAB的roundn功能：按10^decimals的精度四舍五入"""
    if decimals == 0:
        return np.round(x)
    # 核心修正：scale = 10^(-decimals)，将x放大后四舍五入，再缩小回原量级
    scale = 10 ** (-decimals)  
    return np.round(x * scale) / scale

def write_pulse_json(parameters, amp, Pulse_Position, outputfile, fval, 
                     pulse_width_H, pulse_width_P, user="Zidong Lin"):
    """
    生成指定格式的JSON脉冲参数文件
    """
    # 1. 解析参数（逻辑与spinq一致，复用参数处理）
    t_len = len(parameters) // 2
    t_params = parameters[:t_len]
    theta_params = parameters[t_len:]
    
    # 处理参数（时间→微秒，角度→角度制）
    t_params[t_params < 0] = 0.0
    t_params = roundn(t_params, -8)
    t_us = t_params * 1e6  # 时间：秒→微秒（JSON中width用微秒）
    
    theta_deg = theta_params / (2 * np.pi) * 360  # 角度制相位
    theta_deg = roundn(theta_deg, -4)
    
    amp_list = np.array(amp) * 100  # 振幅×100
    total_time = np.sum(t_params)   # 总时间（秒，JSON中用科学计数法）
    slices = len(t_us)              # 脉冲个数
    
    # 2. 构造description字段
    current_date = datetime.now().strftime('%d-%b-%Y')  # 匹配示例格式：09-Sep-2024
    description = {
        "TITLE": outputfile,
        "OWNER": user,
        "DATE": current_date,
        "FIDELITY": f"{-fval/1e7:.6e}",  # 匹配示例量级：9.998651e-01（需除以1e7）
        "TOTALPULSEWIDTH": f"{total_time:.6e}",
        "SLICES": str(slices)  # 示例中SLICES是字符串，保持一致
    }
    
    # 3. 构造pulse字段（区分channel1和channel2）
    channel1_pulse = []  # Pulse_Position=1对应channel1
    channel2_pulse = []  # Pulse_Position=2对应channel2
    
    for i in range(slices):
        # 基础参数（所有脉冲共享）
        width = round(t_us[i], 2)  # 保留2位小数，匹配示例
        detuning = 0               # 示例中detuning均为0
        current_amp = amp_list[i] if i < len(amp_list) else 0
        current_pos = Pulse_Position[i] if i < len(Pulse_Position) else 0
        current_phase = round(theta_deg[i], 4) if i < len(theta_deg) else 0
        
        # 4. 填充channel1（仅当Pulse_Position=1时振幅为current_amp，否则为0）
        if current_pos == 1:
            channel1 = {
                "detuning": detuning,
                "phase": current_phase,
                "amplitude": round(current_amp, 2),
                "width": width
            }
        else:
            channel1 = {
                "detuning": detuning,
                "phase": 0,  # 非当前通道相位设为0
                "amplitude": 0,  # 非当前通道振幅设为0
                "width": width
            }
        
        # 5. 填充channel2（仅当Pulse_Position=2时振幅为current_amp，否则为0）
        if current_pos == 2:
            channel2 = {
                "detuning": detuning,
                "phase": current_phase,
                "amplitude": round(current_amp, 2),
                "width": width
            }
        else:
            channel2 = {
                "detuning": detuning,
                "phase": 0,
                "amplitude": 0,
                "width": width
            }
        
        channel1_pulse.append(channel1)
        channel2_pulse.append(channel2)
    
    # 6. 构造完整JSON结构
    json_data = {
        "description": description,
        "pulse": {
            "channel1_pulse": channel1_pulse,
            "channel2_pulse": channel2_pulse
        }
    }
    
    # 7. 写入JSON文件（缩进2空格，匹配示例格式）
    json_filename = outputfile.replace(".spinq", ".json")  # 替换后缀为.json
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"JSON脉冲参数文件已生成: {json_filename}")
    return json_filename

def write_pulse_parameters(parameters, amp, Pulse_Position, outputfile, fval, 
                          pulse_width_H, pulse_width_P, user="Zidong Lin"):
    """
    将脉冲参数写入指定格式的.spinq文件（修复1维数组索引问题）
    """
    # 核心修复：从1维参数数组拆分时间和角度（t_len = 脉冲个数，时间/角度各占t_len个元素）
    t_len = len(parameters) // 2  # 脉冲个数 = 总参数数 / 2（时间+角度）
    t_params = parameters[:t_len]  # 前t_len个元素：时间参数
    theta_params = parameters[t_len:]  # 后t_len个元素：角度参数（弧度制）
    
    # 处理参数四舍五入
    t_params[t_params < 0] = 0.0  # 关键：先处理负时间
    t_params = roundn(t_params, -8)  # 时间保留到1e-8秒
    # 角度：弧度→角度制四舍五入→转回弧度
    theta_deg = theta_params / (2 * np.pi) * 360  # 弧度→角度
    theta_deg = roundn(theta_deg, -4)  # 角度保留4位小数（如76.2250度）
    theta_params = theta_deg / 360 * 2 * np.pi  # 角度→弧度（供后续计算）
    
    # 单位转换（秒→微秒，弧度→角度制，用于写入文件）
    t = t_params * 1e6  # 时间：秒→微秒（us）
    theta = theta_params / (2 * np.pi) * 360  # 角度：弧度→角度制
    total_time = np.sum(t) * 1e-6  # 总时间（转回秒，用于文件头）
    
    # 准备写入数据（转为列向量格式）
    t_writein = t.reshape(-1, 1)
    theta_writein = theta.reshape(-1, 1)
    amp_writenin = (np.array(amp) * 100).reshape(-1, 1)  # 振幅×100（格式要求）
    
    # 获取当前日期时间
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now()
    hour, minute = current_time.hour, current_time.minute
    
    # 写入文件
    with open(outputfile, 'w') as shpfile:
        # 文件头信息
        shpfile.write(f'##TITLE= {outputfile}\n')
        shpfile.write('##DATA TYPE= Hard Pulse Parameters\n')
        shpfile.write('##ORIGIN= SpinQ Pulses \n')
        shpfile.write(f'##OWNER= {user}\n')
        shpfile.write(f'##DATE= {current_date}\n')
        shpfile.write(f'##GATE_FIDELITY= {-fval:.6f}\n')
        shpfile.write(f'##TIME= {hour}:{minute}\n')
        shpfile.write(f'##TOTALPULSEWIDTH= {total_time:.6e}\n')
        shpfile.write(f'##H_90_PULSEWIDTH= {np.mean(pulse_width_H):.6e}\n')
        shpfile.write(f'##P_90_PULSEWIDTH= {np.mean(pulse_width_P):.6e}\n')
        shpfile.write(f'##SLICES= {len(t_writein)}\n')
        
        # 写入脉冲参数（循环时确保Pulse_Position长度匹配）
        for ct in range(len(t_writein)):
            # 处理Pulse_Position长度不足的情况（避免索引越界）
            pos = Pulse_Position[ct] if (ct < len(Pulse_Position)) else 0
            t_val = t_writein[ct][0]
            theta_val = theta_writein[ct][0]
            amp_val = amp_writenin[ct][0] if (ct < len(amp_writenin)) else 0
            shpfile.write(f'  {pos:7.6e},  {t_val:7.6e},  {theta_val:7.6e},  {amp_val:7.6e}\n')
        
        shpfile.write('##END=\n')
    
    print(f"脉冲参数文件已生成: {outputfile}")
    return outputfile

# 工具函数：张量积
def mykron(*args: np.ndarray) -> np.ndarray:
    """
    通用张量积（Kronecker乘积）函数：支持任意数量的二维方阵输入
    """
    if not args:
        raise ValueError("张量积计算至少需要1个二维方阵输入（当前无输入）")
    
    for idx, mat in enumerate(args):
        if mat.ndim != 2:
            raise ValueError(
                f"第{idx+1}个输入不合法：需为二维方阵，实际维度为{mat.ndim}（形状：{mat.shape}）"
            )
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(
                f"第{idx+1}个输入不合法：需为方阵，实际形状为{mat.shape}（行数≠列数）"
            )
    
    processed_mats = []
    for mat in args:
        complex_mat = mat.astype(np.complex128)
        processed_mats.append(complex_mat)
    
    result = processed_mats[0]
    for mat in processed_mats[1:]:
        result = np.kron(result, mat)
    
    return result

# 获取部分迹
def partial_trace(rho, keep, dims):
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    N = len(dims)
    other = np.setdiff1d(np.arange(N), keep)
    
    # 关键优化：直接计算迹，不创建冗余的高维重塑数组
    # 计算保留维度和迹掉维度的尺寸
    keep_size = np.prod(dims[keep])
    other_size = np.prod(dims[other])
    
    # 重塑为 (keep_size, other_size, keep_size, other_size)，直接对other_size维度求和
    rho_reshaped = rho.reshape((keep_size, other_size, keep_size, other_size))
    traced_rho = np.einsum('abac->bc', rho_reshaped)  # 仅保留keep维度，求和other维度
    
    return traced_rho

def matlab_fidelity(rho1, rho2):
    """
    复现MATLAB的Fidelity函数：计算两个密度矩阵的保真度
    公式：F(rho1, rho2) = [tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))]^2
    rho1, rho2: 输入密度矩阵（需确保为Hermitian矩阵）
    return: 保真度（0~1）
    """
    # 确保密度矩阵归一化（MATLAB中默认输入已归一化，此处冗余处理）
    rho1 = rho1 / np.trace(rho1) if np.trace(rho1) != 0 else rho1
    rho2 = rho2 / np.trace(rho2) if np.trace(rho2) != 0 else rho2
    
    # 计算sqrt(rho1)
    sqrt_rho1 = sqrtm(rho1)
    # 计算中间矩阵：sqrt(rho1)*rho2*sqrt(rho1)
    intermediate = sqrt_rho1 @ rho2 @ sqrt_rho1
    # 计算中间矩阵的平方根（处理数值误差导致的非Hermitian问题）
    sqrt_intermediate = sqrtm(intermediate.astype(np.complex128))
    # 计算迹的绝对值并平方
    return np.abs(np.trace(sqrt_intermediate)) ** 2

def one_click_smp_fidelity(parameters, U_theo, local_vars):
    pulse_config = {
        # 11-12：P门（5个脉冲，无固定项）
        11: [5, [1]*5, []],
        12: [5, [1]*5, []],
        # 13-14：10个脉冲（13=H门，14=P门，无固定项）
        13: [10, [0]*10, []],
        14: [10, [1]*10, []],
        # 15：CNOT12（31个脉冲，第6个是固定H_int{2}，1-5=P，7-31=H）
        15: [31, [1]*5 + [2] + [0]*25, [5]],  # 固定项在Python索引5（MATLAB6）
        # 16：CNOT21（31个脉冲，第6个是固定H_int{2}，1-5=H，7-31=P）
        16: [31, [0]*5 + [2] + [1]*25, [5]],
        # 17：CZ（31个脉冲，第1个是固定H_int{2}，2-6=P，7-31=H）
        17: [31, [2] + [1]*5 + [0]*25, [0]],
        # 18：SWAP（93个脉冲，无固定项，H/P交替，按MATLAB顺序映射）
        18: [93, [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*5 + [0]*5 + [1]*3, []],
        # 19：PPS_PART1（26个脉冲，第9、18个是固定H_int{2}，1-4=P，5-8=H，10-13=P，14-17=H，19-22=P，23-26=H）
        19: [26, [1]*4 + [0]*4 + [2] + [1]*4 + [0]*4 + [2] + [1]*4 + [0]*4, [8, 17]],
        # 20：PPS_PART2（22个脉冲，第6、17个是固定H_int{2}，1-5=P，7-11=H，12-16=P，18-22=H）
        20: [22, [1]*5 + [2] + [0]*5 + [1]*5 + [2] + [0]*5, [5, 16]]
    }
    # 补充1-10号脉冲配置（原有逻辑）
    for p in range(1, 11):
        if p in [1,2,3,4,9,10]:  # H门（5个脉冲）
            pulse_config[p] = [5, [0]*5, []]
        else:  # P门（5个脉冲）
            pulse_config[p] = [5, [1]*5, []]

    # -------------------------- 2. 初始化基础参数 --------------------------
    pulsenumber = local_vars.pulsenumber
    if pulsenumber not in pulse_config:
        raise ValueError(f"不支持脉冲编号：{pulsenumber}（仅1-20）")
    
    t_len, ham_map, fixed_h_pos = pulse_config[pulsenumber]
    H_int = local_vars.H_int
    num_h_int = len(H_int)
    num_pulse_h = len(local_vars.H_pulseHx)  # H/P脉冲列表长度一致
    f_matrix = np.zeros((num_h_int, num_pulse_h), dtype=np.float64)

    # 拆分参数：前t_len个为时间，后t_len个为相位（MATLAB parameters(1,:)和(2,:)）
    t_list = parameters[:t_len]
    theta_list = parameters[t_len:]
    # MATLAB幺正累积顺序：从最后一个脉冲乘到第一个（逆序处理）
    t_list_rev = t_list[::-1]
    theta_list_rev = theta_list[::-1]
    # 逆序后，固定H_int{2}的位置也需调整（原位置→逆序后位置）
    fixed_h_pos_rev = [t_len - 1 - pos for pos in fixed_h_pos]

    # -------------------------- 3. 循环计算U_SMP与保真度 --------------------------
    for ii in range(num_h_int):
        h = H_int[ii]  # 当前相互作用哈密顿量（MATLAB H_int{ii}）
        # 固定H_int{2}：MATLAB H_int{2}→Python H_int[1]（1-based→0-based）
        h_fixed = H_int[1] if len(H_int) >= 2 else h

        for jj in range(num_pulse_h):
            # 初始化幺正变换为单位矩阵
            U_smp = np.eye(U_theo.shape[0], dtype=np.complex128)

            # -------------------------- 3.1 累积幺正变换（核心：处理固定H_int项） --------------------------
            for idx in range(t_len):
                current_t = t_list_rev[idx]
                current_theta = theta_list_rev[idx] if idx not in fixed_h_pos_rev else 0
                current_ham_type = ham_map[::-1][idx]  # 逆序后匹配ham_map

                # 分支1：固定H_int{2}项（无脉冲哈密顿量，仅H_int{2}）
                if current_ham_type == 2:
                    U_pulse = expm(-1j * h_fixed * current_t)
                
                # 分支2：H门（H_pulseHx/Hy）
                elif current_ham_type == 0:
                    pulse_hx = local_vars.H_pulseHx[jj] * np.cos(current_theta)
                    pulse_hy = local_vars.H_pulseHy[jj] * np.sin(current_theta)
                    total_h = h + pulse_hx + pulse_hy
                    U_pulse = expm(-1j * total_h * current_t)
                
                # 分支3：P门（H_pulsePx/Py）
                elif current_ham_type == 1:
                    pulse_hx = local_vars.H_pulsePx[jj] * np.cos(current_theta)
                    pulse_hy = local_vars.H_pulsePy[jj] * np.sin(current_theta)
                    total_h = h + pulse_hx + pulse_hy
                    U_pulse = expm(-1j * total_h * current_t)
                
                # 累积幺正变换（MATLAB右乘顺序）
                U_smp = U_pulse @ U_smp

            # -------------------------- 3.2 保真度计算（扩展多类型初始态/部分迹） --------------------------
            if local_vars.optim_type == 1:
                # 类型1：幺正保真度（所有脉冲统一逻辑）
                trace_val = np.trace(np.conj(U_smp).T @ U_theo)
                fid = np.abs(trace_val) / U_theo.shape[0]
                f_matrix[ii, jj] = -1e7 * fid

            elif local_vars.optim_type == 2:
                # 类型2：状态保真度（按脉冲切换初始态/部分迹）
                f_state = 0.0
                num_state = len(local_vars.state_theo)

                # 按脉冲选择初始态
                if pulsenumber in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
                    init_rho_list = local_vars.rho0_6  # 1-18号用rho0_6
                elif pulsenumber == 19:
                    init_rho_list = local_vars.rho0_PPS1_3  # 19号type2用rho0_PPS1_3
                elif pulsenumber == 20:
                    init_rho_list = local_vars.rho0_3  # 20号用rho0_3
                else:
                    init_rho_list = [np.eye(U_smp.shape[0])/U_smp.shape[0]]  # 默认

                # 循环计算每个初始态的保真度
                for kk in range(num_state):
                    rho0 = init_rho_list[kk] if kk < len(init_rho_list) else init_rho_list[0]
                    rho_theo = local_vars.state_theo[kk]
                    rho_evolved = U_smp @ rho0 @ np.conj(U_smp).T

                    # 按脉冲选择部分迹参数
                    if pulsenumber in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
                        # 保留[3-8]（MATLAB1-based）→[2-7]（Python0-based），8量子比特
                        rho_smp = partial_trace(rho_evolved, keep=[2,3,4,5,6,7], dims=[2]*8)
                    elif pulsenumber == 19:
                        # 19号无部分迹（直接用演化后态）
                        rho_smp = rho_evolved
                    elif pulsenumber == 20:
                        # 20号保留[3]（MATLAB1-based）→[2]（Python0-based），3量子比特
                        rho_smp = partial_trace(rho_evolved, keep=[2], dims=[2,2,2])
                    else:
                        rho_smp = rho_evolved

                    # 按MATLAB选择保真度计算方式（1-norm或Fidelity）
                    if pulsenumber in [2,3,4,6,7,8,10,18]:
                        # 用1-norm（MATLAB注释外逻辑）
                        fid_state = 1 - np.linalg.norm(rho_smp - rho_theo)
                    else:
                        # 用Fidelity（MATLAB注释内逻辑）
                        fid_state = matlab_fidelity(rho_smp, rho_theo)

                    f_state += fid_state

                # 平均状态保真度（乘-1e7）
                avg_f_state = f_state / num_state
                f_matrix[ii, jj] = -1e7 * avg_f_state

            elif local_vars.optim_type == 3:
                # 类型3：19号脉冲专属（用rho0_PPS1_8，无部分迹）
                if pulsenumber != 19:
                    raise ValueError("仅19号脉冲支持optim_type=3")
                
                f_state = 0.0
                num_state = len(local_vars.state_theo)
                init_rho_list = local_vars.rho0_PPS1_8

                for kk in range(num_state):
                    rho0 = init_rho_list[kk] if kk < len(init_rho_list) else init_rho_list[0]
                    rho_theo = local_vars.state_theo[kk]
                    rho_evolved = U_smp @ rho0 @ np.conj(U_smp).T
                    fid_state = matlab_fidelity(rho_evolved, rho_theo)
                    f_state += fid_state

                avg_f_state = f_state / num_state
                f_matrix[ii, jj] = -1e7 * avg_f_state

    # -------------------------- 4. 返回平均保真度（匹配MATLAB mean(mean(f))） --------------------------
    mean_f = np.mean(f_matrix)
    return mean_f

local_vars = GlobalVars()

# 脉冲优化线程函数
def optimize_pulse(params_queue, result_queue, fit_params, pulse_params, local_vars):
    try:
        hbar = (6.626e-34) / (2 * np.pi)  # 约化普朗克常数
        KB = 1.3807e-23
        (PW_H, PW_P, T_indoor, w_H, w_P, J12, J23, 
         w_1, w_2, w_3, w_4, o1, o2, pulse_width_H, pulse_width_P, File_Type) = fit_params
        pulsenumber, outputfile, threshold = pulse_params
        I = local_vars.I
        Ix, Iy, Iz = local_vars.Ix, local_vars.Iy, local_vars.Iz
        X, Y, Z = local_vars.X, local_vars.Y, local_vars.Z
        Hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex64)
        sqrtX = (1 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]], dtype=np.complex64)  # X门平方根
        sqrtY = (1 / np.sqrt(2)) * np.array([[1, -1], [1, 1]], dtype=np.complex64)  # Y门平方根
        sqrtZ = (1 / np.sqrt(2)) * np.array([[1-1j, 0], [0, 1+1j]], dtype=np.complex64)  # Z门平方根

        # 初始化存储哈密顿量的字典列表
        H_int_2 = []
        H_pulseHy_2 = []
        H_pulseHx_2 = []
        H_pulsePy_2 = []
        H_pulsePx_2 = []

        # 循环计算2自旋体系的哈密顿量
        for ii in range(len(o1)):
            term1 = 2 * np.pi * ((w_1 - o1[ii]) * mykron(Iz, I))
            term2 = 2 * np.pi * ((w_2 - o2[ii]) * mykron(I, Iz))
            term3 = 2 * np.pi * J12 * mykron(Iz, Iz)
            H_int_2.append(term1 + term2 + term3)
            H_pulseHy_2.append(2 * np.pi / (pulse_width_H[ii] * 4) * mykron(Iy, I))
            H_pulseHx_2.append(2 * np.pi / (pulse_width_H[ii] * 4) * mykron(Ix, I))
            H_pulsePy_2.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Iy))
            H_pulsePx_2.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Ix))

        # 初始化3自旋体系的哈密顿量存储列表
        H_int_3 = []
        H_pulseHy_3 = []
        H_pulseHx_3 = []
        H_pulsePy_3 = []
        H_pulsePx_3 = []

        # 循环计算3自旋体系的哈密顿量
        for ii in range(len(o1)):
            term1 = 2 * np.pi * ((w_1 - o1[ii]) * mykron(Iz, I, I))
            term2 = 2 * np.pi * ((w_2 - o2[ii]) * mykron(I, Iz, I))
            term3 = 2 * np.pi * ((w_3 - o1[ii]) * mykron(I, I, Iz))
            term4 = 2 * np.pi * J12 * mykron(Iz, Iz, I)
            term5 = 2 * np.pi * J23 * mykron(I, Iz, Iz)
            H_int_3.append(term1 + term2 + term3 + term4 + term5)
            pulse_hy = 2 * np.pi / (pulse_width_H[ii] * 4) * (mykron(Iy, I, I) + mykron(I, I, Iy))
            H_pulseHy_3.append(pulse_hy)
            pulse_hx = 2 * np.pi / (pulse_width_H[ii] * 4) * (mykron(Ix, I, I) + mykron(I, I, Ix))
            H_pulseHx_3.append(pulse_hx)
            H_pulsePy_3.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Iy, I))
            H_pulsePx_3.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Ix, I))

        # 初始化4自旋体系的哈密顿量存储列表
        H_int_4 = []
        H_pulseHy_4 = []
        H_pulseHx_4 = []
        H_pulsePy_4 = []
        H_pulsePx_4 = []

        # 循环计算4自旋体系的哈密顿量
        for ii in range(len(o1)):
            term1 = 2 * np.pi * ((w_1 - o1[ii]) * mykron(Iz, I, I, I))
            term2 = 2 * np.pi * ((w_2 - o2[ii]) * mykron(I, Iz, I, I))
            term3 = 2 * np.pi * ((w_3 - o1[ii]) * mykron(I, I, Iz, I))
            term4 = 2 * np.pi * ((w_4 - o1[ii]) * mykron(I, I, I, Iz))
            term5 = 2 * np.pi * J12 * mykron(Iz, Iz, I, I)
            term6 = 2 * np.pi * J23 * mykron(I, Iz, Iz, I)
            H_int_4.append(term1 + term2 + term3 + term4 + term5 + term6)
            pulse_hy = 2 * np.pi / (pulse_width_H[ii] * 4) * (
                mykron(Iy, I, I, I) + mykron(I, I, Iy, I) + mykron(I, I, I, Iy)
            )
            H_pulseHy_4.append(pulse_hy)
            pulse_hx = 2 * np.pi / (pulse_width_H[ii] * 4) * (
                mykron(Ix, I, I, I) + mykron(I, I, Ix, I) + mykron(I, I, I, Ix)
            )
            H_pulseHx_4.append(pulse_hx)     
            H_pulsePy_4.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Iy, I, I))
            H_pulsePx_4.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Ix, I, I))

        # 初始化6自旋体系的哈密顿量存储列表
        H_int_6 = []
        H_pulseHy_6 = []
        H_pulseHx_6 = []
        H_pulsePy_6 = []
        H_pulsePx_6 = []

        # 循环计算6自旋体系的哈密顿量
        for ii in range(len(o1)):
            term1 = 2 * np.pi * ((w_1 - o1[ii]) * mykron(Iz, I, I, I, I, I, I, I))
            term2 = 2 * np.pi * ((w_2 - o2[ii]) * mykron(I, Iz, I, I, I, I, I, I))
            term3 = 2 * np.pi * (w_3 - o1[ii]) * (
                mykron(I, I, Iz, I, I, I, I, I) +
                mykron(I, I, I, Iz, I, I, I, I) +
                mykron(I, I, I, I, Iz, I, I, I) +
                mykron(I, I, I, I, I, Iz, I, I) +
                mykron(I, I, I, I, I, I, Iz, I) +
                mykron(I, I, I, I, I, I, I, Iz)
            )
            term4 = 2 * np.pi * J12 * mykron(Iz, Iz, I, I, I, I, I, I)
            term5 = 2 * np.pi * J23 * (
                mykron(I, Iz, Iz, I, I, I, I, I) +
                mykron(I, Iz, I, Iz, I, I, I, I) +
                mykron(I, Iz, I, I, Iz, I, I, I) +
                mykron(I, Iz, I, I, I, Iz, I, I) +
                mykron(I, Iz, I, I, I, I, Iz, I) +
                mykron(I, Iz, I, I, I, I, I, Iz)
            )
            H_int_6.append(term1 + term2 + term3 + term4 + term5)            
            pulse_hy = 2 * np.pi / (pulse_width_H[ii] * 4) * (
                mykron(Iy, I, I, I, I, I, I, I) +
                mykron(I, I, Iy, I, I, I, I, I) +
                mykron(I, I, I, Iy, I, I, I, I) +
                mykron(I, I, I, I, Iy, I, I, I) +
                mykron(I, I, I, I, I, Iy, I, I) +
                mykron(I, I, I, I, I, I, Iy, I) +
                mykron(I, I, I, I, I, I, I, Iy)
            )
            H_pulseHy_6.append(pulse_hy)
            pulse_hx = 2 * np.pi / (pulse_width_H[ii] * 4) * (
                mykron(Ix, I, I, I, I, I, I, I) +
                mykron(I, I, Ix, I, I, I, I, I) +
                mykron(I, I, I, Ix, I, I, I, I) +
                mykron(I, I, I, I, Ix, I, I, I) +
                mykron(I, I, I, I, I, Ix, I, I) +
                mykron(I, I, I, I, I, I, Ix, I) +
                mykron(I, I, I, I, I, I, I, Ix)
            )
            H_pulseHx_6.append(pulse_hx)
            H_pulsePy_6.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Iy, I, I, I, I, I, I))
            H_pulsePx_6.append(2 * np.pi / (pulse_width_P[ii] * 4) * mykron(I, Ix, I, I, I, I, I, I))

        # 定义量子门
        def define_quantum_gates():
            GHZ = 0.5 * np.outer(np.array([1, 0, 0, 1], dtype=np.complex128).conj().T, np.array([1, 0, 0, 1], dtype=np.complex128))
            T = np.array([[1, 0], [0, (1+1j)/np.sqrt(2)]], dtype=np.complex128)  # T门
            Td = np.array([[1, 0], [0, (1-1j)/np.sqrt(2)]], dtype=np.complex128)  # T†门（T的共轭转置）

            X1 = mykron(X, I)
            X2 = mykron(I, X)
            Y1 = mykron(Y, I)
            Y2 = mykron(I, Y)
            Z1 = mykron(Z, I)
            Z2 = mykron(I, Z)
            sqrtX1 = mykron(sqrtX, I)
            sqrtX2 = mykron(I, sqrtX)
            sqrtY1 = mykron(sqrtY, I)
            sqrtY2 = mykron(I, sqrtY)
            sqrtZ1 = mykron(sqrtZ, I)
            sqrtZ2 = mykron(I, sqrtZ)
            T1 = mykron(T, I)
            T2 = mykron(I, T)
            Td1 = mykron(Td, I)
            Td2 = mykron(I, Td)
            I1 = mykron(I, I)  # 2自旋单位算符（冗余定义，与I2一致）
            I2 = mykron(I, I)  # 2自旋单位算符

            # 受控门（CNOT/CZ）
            CNOT12 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), X)  # 控制Q1，目标Q2
            CNOT21 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + mykron(X, np.array([[0, 0], [0, 1]], dtype=np.complex128))  # 控制Q2，目标Q1
            CZ12 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), Z)  # 控制Q1，目标Q2
            CZ21 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + mykron(Z, np.array([[0, 0], [0, 1]], dtype=np.complex128))  # 控制Q2，目标Q1

            # 交换门与置换门
            SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)  # 2自旋SWAP门
            local_vars.U_permute = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.complex128)  # 2自旋置换门

            # J耦合演化门
            U_J = expm(-1j * 2 * np.pi * J12 * mykron(Iz, Iz) * (1 / (2 * J12)))

            # 量子门列表（对应MATLAB的 U_theo 细胞数组，按原顺序排列）
            local_vars.U_theo.append(sqrtX1)  # 1: Q1的sqrtX门
            local_vars.U_theo.append(sqrtY1)  # 2: Q1的sqrtY门
            local_vars.U_theo.append(mykron(sqrtX.conj().T, I))  # 3: Q1的sqrtX†门（共轭转置）
            local_vars.U_theo.append(mykron(sqrtY.conj().T, I))  # 4: Q1的sqrtY†门（共轭转置）
            local_vars.U_theo.append(sqrtX2)  # 5: Q2的sqrtX门
            local_vars.U_theo.append(sqrtY2)  # 6: Q2的sqrtY门
            local_vars.U_theo.append(mykron(I, sqrtX.conj().T))  # 7: Q2的sqrtX†门（共轭转置）
            local_vars.U_theo.append(mykron(I, sqrtY.conj().T))  # 8: Q2的sqrtY†门（共轭转置）
            local_vars.U_theo.append(mykron(X, I))  # 9: Q1的X门
            local_vars.U_theo.append(mykron(Y, I))  # 10: Q1的Y门
            local_vars.U_theo.append(mykron(I, X))  # 11: Q2的X门
            local_vars.U_theo.append(mykron(I, Y))  # 12: Q2的Y门
            local_vars.U_theo.append(mykron(Hadamard, I))  # 13: Q1的Hadamard门
            local_vars.U_theo.append(mykron(I, Hadamard))  # 14: Q2的Hadamard门
            local_vars.U_theo.append(CNOT12)  # 15: CNOT12门
            local_vars.U_theo.append(CNOT21)  # 16: CNOT21门
            local_vars.U_theo.append(CZ12)  # 17: CZ12门
            local_vars.U_theo.append(SWAP)  # 18: SWAP门

            half_I = 0.5 * I
            rhoX = half_I + Ix  # X方向极化密度矩阵
            rhoY = half_I + Iy  # Y方向极化密度矩阵
            rhoZ = half_I + Iz  # Z方向极化密度矩阵
            rho = [
                half_I,  # 1: 单自旋混合态（1/2 I）
                rhoX,     # 2: X极化态
                rhoY,     # 3: Y极化态
                rhoZ      # 4: Z极化态
            ]

            local_vars.rho0_6.append(mykron(rhoX, half_I, rhoX, rhoX, rhoX, rhoX, rhoX, rhoX)) 
            local_vars.rho0_6.append(mykron(rhoX, rhoX, rhoX, rhoX, rhoX, rhoX, rhoX, rhoX))   
            local_vars.rho0_6.append(mykron(rhoX, rhoY, rhoX, rhoX, rhoX, rhoX, rhoX, rhoX))   
            local_vars.rho0_6.append(mykron(rhoX, rhoZ, rhoX, rhoX, rhoX, rhoX, rhoX, rhoX))   
            local_vars.rho0_6.append(mykron(rhoY, half_I, rhoY, rhoY, rhoY, rhoY, rhoY, rhoY)) 
            local_vars.rho0_6.append(mykron(rhoY, rhoX, rhoY, rhoY, rhoY, rhoY, rhoY, rhoY))   
            local_vars.rho0_6.append(mykron(rhoY, rhoY, rhoY, rhoY, rhoY, rhoY, rhoY, rhoY))   
            local_vars.rho0_6.append(mykron(rhoY, rhoZ, rhoY, rhoY, rhoY, rhoY, rhoY, rhoY))   
            local_vars.rho0_6.append(mykron(rhoZ, half_I, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ)) 
            local_vars.rho0_6.append(mykron(rhoZ, rhoX, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ))   
            local_vars.rho0_6.append(mykron(rhoZ, rhoY, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ))   
            local_vars.rho0_6.append(mykron(rhoZ, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ, rhoZ))   
            local_vars.rho0_6.append(mykron(half_I, half_I, half_I, half_I, half_I, half_I, half_I, half_I))  
            local_vars.rho0_6.append(mykron(half_I, rhoX, half_I, half_I, half_I, half_I, half_I, half_I))    
            local_vars.rho0_6.append(mykron(half_I, rhoY, half_I, half_I, half_I, half_I, half_I, half_I))    
            local_vars.rho0_6.append(mykron(half_I, rhoZ, half_I, half_I, half_I, half_I, half_I, half_I))    

            # # 三重循环补充rho0_6（ii/jj/kk遍历rho[0-3]，共4*4*4=64个，对应MATLAB循环）
            # count = 16  # 前16个已添加，从索引16开始（Python列表0-based）
            # for ii in range(4):
            #     for jj in range(4):
            #         for kk in range(4):
            #             # 对应MATLAB: mykron(rho{ii},rho{jj},rho{kk},rho{kk},rho{kk},rho{kk},rho{kk},rho{kk})
            #             rho_6 = mykron(
            #                 rho[ii], rho[jj], rho[kk],
            #                 rho[kk], rho[kk], rho[kk],
            #                 rho[kk], rho[kk]
            #             )
            #             local_vars.rho0_6.append(rho_6)
            #             count += 1

            local_vars.rho0_3.append(mykron(rhoX, half_I, rhoX))  # 1
            local_vars.rho0_3.append(mykron(rhoX, rhoX, rhoX))    # 2
            local_vars.rho0_3.append(mykron(rhoX, rhoY, rhoX))    # 3
            local_vars.rho0_3.append(mykron(rhoX, rhoZ, rhoX))    # 4
            local_vars.rho0_3.append(mykron(rhoY, half_I, rhoY))  # 5
            local_vars.rho0_3.append(mykron(rhoY, rhoX, rhoY))    # 6
            local_vars.rho0_3.append(mykron(rhoY, rhoY, rhoY))    # 7
            local_vars.rho0_3.append(mykron(rhoY, rhoZ, rhoY))    # 8
            local_vars.rho0_3.append(mykron(rhoZ, half_I, rhoZ))  # 9
            local_vars.rho0_3.append(mykron(rhoZ, rhoX, rhoZ))    # 10
            local_vars.rho0_3.append(mykron(rhoZ, rhoY, rhoZ))    # 11
            local_vars.rho0_3.append(mykron(rhoZ, rhoZ, rhoZ))    # 12
            local_vars.rho0_3.append(mykron(half_I, half_I, half_I))  # 13（全混合）
            local_vars.rho0_3.append(mykron(half_I, rhoX, half_I))    # 14
            local_vars.rho0_3.append(mykron(half_I, rhoY, half_I))    # 15
            local_vars.rho0_3.append(mykron(half_I, rhoZ, half_I))    # 16

            # 三重循环补充rho0_3（对应MATLAB循环）
            # count = 16
            # for ii in range(4):
            #     for jj in range(4):
            #         for kk in range(4):
            #             # 对应MATLAB: mykron(rho{ii},rho{jj},rho{kk})
            #             rho_3 = mykron(rho[ii], rho[jj], rho[kk])
            #             local_vars.rho0_3.append(rho_3)
            #             count += 1

            local_vars.rho0_4.append(mykron(rhoX, half_I, rhoX, rhoX))  # 1
            local_vars.rho0_4.append(mykron(rhoX, rhoX, rhoX, rhoX))    # 2
            local_vars.rho0_4.append(mykron(rhoX, rhoY, rhoX, rhoX))    # 3
            local_vars.rho0_4.append(mykron(rhoX, rhoZ, rhoX, rhoX))    # 4
            local_vars.rho0_4.append(mykron(rhoY, half_I, rhoY, rhoY))  # 5
            local_vars.rho0_4.append(mykron(rhoY, rhoX, rhoY, rhoY))    # 6
            local_vars.rho0_4.append(mykron(rhoY, rhoY, rhoY, rhoY))    # 7
            local_vars.rho0_4.append(mykron(rhoY, rhoZ, rhoY, rhoY))    # 8
            local_vars.rho0_4.append(mykron(rhoZ, half_I, rhoZ, rhoZ))  # 9
            local_vars.rho0_4.append(mykron(rhoZ, rhoX, rhoZ, rhoZ))    # 10
            local_vars.rho0_4.append(mykron(rhoZ, rhoY, rhoZ, rhoZ))    # 11
            local_vars.rho0_4.append(mykron(rhoZ, rhoZ, rhoZ, rhoZ))    # 12
            local_vars.rho0_4.append(mykron(half_I, half_I, half_I, half_I))  # 13（全混合）
            local_vars.rho0_4.append(mykron(half_I, rhoX, half_I, half_I))    # 14
            local_vars.rho0_4.append(mykron(half_I, rhoY, half_I, half_I))    # 15
            local_vars.rho0_4.append(mykron(half_I, rhoZ, half_I, half_I))    # 16

            return rho
        
        # 获取量子门
        rho = define_quantum_gates()

        # 计算热平衡参数
        sigma_H = hbar * w_H / (KB * T_indoor)
        sigma_P = hbar * w_P / (KB * T_indoor)
        
        # 平衡密度矩阵
        rho_eq_H = 0.5 * np.array([[1 + sigma_H/2, 0], [0, 1 - sigma_H/2]])
        rho_eq_P = 0.5 * np.array([[1 + sigma_P/2, 0], [0, 1 - sigma_P/2]])
        rho_eq = mykron(rho_eq_H, rho_eq_P)
        min_diag = np.min(np.diag(rho_eq))
        rho_eff = rho_eq - min_diag * mykron(I, I)

        # 计算线选幺正变换 U_lineselective
        term1 = (56/360) * 2 * np.pi * mykron(0.5 * (I - 2 * Iz), Ix)
        term2 = (266.5/360) * 2 * np.pi * mykron(Ix, 0.5 * (I - 2 * Iz))
        U_lineselective = expm(-1j * (term1 + term2))

        # 计算H完全弛豫后的线选幺正变换 U_lineselective_PostRelaxation
        term1_post = (120.7371/360) * 2 * np.pi * mykron(0.5 * (I - 2 * Iz), Ix)
        term2_post = (164.9300/360) * 2 * np.pi * mykron(Ix, 0.5 * (I - 2 * Iz))
        U_lineselective_PostRelaxation = expm(-1j * (term1_post + term2_post))

        # 计算 rho_line = U_lineselective * rho_eq * U_lineselective'
        rho_line = U_lineselective @ rho_eq @ U_lineselective.conj().T

        # 初始化 rho0_PPS1 系列
        rho0_PPS1 = {}

        # 计算 rho0_PPS1_PostRelaxation{1}
        kron_ip = mykron(I, rho_eq_P)
        min_diag_ip = np.min(np.diag(kron_ip))
        local_vars.rho0_PPS1_PostRelaxation = {
            1: 1e7 * (kron_ip - min_diag_ip * mykron(I, I))
        }

        # 计算 rho0_PPS1_3{1}（3自旋系统）
        kron_hph = mykron(rho_eq_H, rho_eq_P, rho_eq_H)
        min_diag_hph = np.min(np.diag(kron_hph))
        local_vars.rho0_PPS1_3 = [
            1.0e+05 * (kron_hph - min_diag_hph * mykron(I, I, I))
        ]
        # 计算 rho0_PPS1_8{1}（8自旋系统）
        kron_8spin = mykron(
            rho_eq_H, rho_eq_P, rho_eq_H, rho_eq_H,
            rho_eq_H, rho_eq_H, rho_eq_H, rho_eq_H
        )
        min_diag_8spin = np.min(np.diag(kron_8spin))

        # 8个单位矩阵的克罗内克积
        kron_8I = mykron(I, I, I, I, I, I, I, I)
        local_vars.rho0_PPS1_8 = {
            1: 1.0e+05 * (kron_8spin - min_diag_8spin * kron_8I)
        }

        pulse_results = {}
        local_vars.optim_type = 2
        if pulsenumber == 1:
            U_theo = local_vars.U_theo[0]
            Pulse_Position = np.array([1, 1, 1, 1, 1]).conj().T
            # 加载预定义t/Q1_X90（根据PW_H/PW_P）
            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_Q1_X90 = np.array([6.148000e+01, 9.164000e+01, 1.991900e+02, 6.209000e+01, 3.420000e+00]) * 1e-6
                local_vars.theta_Q1_X90 = np.array([7.539860e+01, 1.168412e+02, 2.978823e+02, 7.639280e+01, 3.604785e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_Q1_X90 = np.array([6.148000e+01, 9.164000e+01, 1.991900e+02, 6.209000e+01, 3.420000e+00]) * 1e-6
                local_vars.theta_Q1_X90 = np.array([7.539860e+01, 1.168412e+02, 2.978823e+02, 7.639280e+01, 3.604785e+02]) / 360 * 2 * np.pi
            elif (PW_H == 30e-6) and (PW_P == 30e-6):
                local_vars.t_Q1_X90 = np.array([9.578000e+01, 5.402000e+01, 7.013000e+01, 8.600000e-01, 4.686000e+01]) * 1e-6
                local_vars.theta_Q1_X90 = np.array([1.643180e+01, 1.215655e+02, 3.722070e+01, 5.257290e+01, 1.085253e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_Q1_X90 = np.array([])
                local_vars.theta_Q1_X90 = np.array([])
            
            # 无预定义参数：随机初始化并优化
            if len(local_vars.t_Q1_X90) == 0:
                local_vars.H_int = H_int_2
                local_vars.H_pulseHx = H_pulseHx_2
                local_vars.H_pulseHy = H_pulseHy_2
                local_vars.H_pulsePx = H_pulsePx_2
                local_vars.H_pulsePy = H_pulsePy_2
                local_vars.optim_type = 1
                fval = 0
                # 循环优化直到保真度达标
                while -fval / 1e7 <= threshold:
                    t = np.array([6.148000e+01, 9.164000e+01, 1.991900e+02, 6.209000e+01, 3.420000e+00]) * 1e-6
                    theta = np.array([7.539860e+01, 1.168412e+02, 2.978823e+02, 7.639280e+01, 3.604785e+02]) / 360 * 2 * np.pi
                    amp = np.ones(len(t))
                    parameters = np.concatenate([t, theta])

                    # 设置边界
                    t_len = len(t)
                    lb_t = np.full(t_len, -0.0000001e-6)
                    lb_theta = np.full(t_len, -100)
                    lb = np.concatenate([lb_t, lb_theta])
                    ub_t = np.full(t_len, 1000e-6)
                    ub_theta = np.full(t_len, 100)
                    ub = np.concatenate([ub_t, ub_theta])
                    bounds = list(zip(lb, ub))
                    
                    # 优化选项
                    options = {
                        'maxiter': 200,
                        'maxfun': 10000,
                        'disp': True,
                        "gtol": 1e-5,
                        "ftol": 1e-8
                    }

                    # 2次优化（原MATLAB for optimize=1:2）
                    for _ in range(2):
                        result = minimize(
                            fun=one_click_smp_fidelity,
                            x0=parameters,
                            args=(U_theo, local_vars),
                            method="L-BFGS-B",
                            bounds=bounds,
                            options=options
                        )
                        parameters = result.x
                        fval = result.fun
                    
                    # 负相位调整（加2π转为正相位）
                    for nn in range(t_len, len(parameters)):
                        if parameters[nn] < 0:
                            parameters[nn] += 2 * np.pi
                    
                    t = parameters[:t_len]
                    theta = parameters[t_len:]
                        
                # 切换到8自旋系统优化（optim_type=2）
                U_theo = mykron(sqrtX, I, I, I, I, I, I, I)  # 8自旋理论幺正变换
                local_vars.H_int = H_int_6
                local_vars.H_pulseHx = H_pulseHx_6
                local_vars.H_pulseHy = H_pulseHy_6
                local_vars.H_pulsePx = H_pulsePx_6
                local_vars.H_pulsePy = H_pulsePy_6
                
                # 计算理论目标状态（原MATLAB state_theo）
                local_vars.state_theo = []
                for kk in range(len(local_vars.rho0_6)):
                    rho_transformed = U_theo @ local_vars.rho0_6[kk] @ U_theo.conj().T
                    rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2,2,2,2,2,2,2,2])
                    local_vars.state_theo.append(rho_traced)
                
                # 8自旋系统优化（2次迭代）
                local_vars.optim_type = 2  
                MaxIter = 100
                options["maxiter"] = MaxIter
                for _ in range(2):
                    result = minimize(
                        fun=one_click_smp_fidelity,
                        x0=parameters,
                        args=(U_theo, local_vars),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=options
                    )
                    parameters = result.x
                    fval = result.fun
                
                # 负相位调整
                for nn in range(t_len, len(parameters)):
                    if parameters[nn] < 0:
                        parameters[nn] += 2 * np.pi
                
                # 更新最终t和theta
                t = parameters[:t_len]
                theta = parameters[t_len:]
            
            # 有预定义参数：直接优化（optim_type=2）
            else:
                local_vars.optim_type = 2
                t=local_vars.t_Q1_X90
                theta=local_vars.theta_Q1_X90
                amp = np.ones(len(t))
                parameters = np.concatenate([t, theta])
                t_len = len(t)
                
                # 8自旋理论幺正变换
                U_theo = mykron(sqrtX, I, I, I, I, I, I, I)
                local_vars.H_int = H_int_6
                local_vars.H_pulseHx = H_pulseHx_6
                local_vars.H_pulseHy = H_pulseHy_6
                local_vars.H_pulsePx = H_pulsePx_6
                local_vars.H_pulsePy = H_pulsePy_6
                
                # 计算理论目标状态
                local_vars.state_theo = []
                for kk in range(len(local_vars.rho0_6)):
                    rho_transformed = U_theo @ local_vars.rho0_6[kk] @ U_theo.conj().T
                    rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2,2,2,2,2,2,2,2])
                    local_vars.state_theo.append(rho_traced)
                
                # 边界约束
                lb_t = np.full(t_len, -0.0000001e-6)
                lb_theta = np.full(t_len, -100)
                lb = np.concatenate([lb_t, lb_theta])
                
                ub_t = np.full(t_len, 1000e-6)
                ub_theta = np.full(t_len, 100)
                ub = np.concatenate([ub_t, ub_theta])
                bounds = list(zip(lb, ub))
                
                # 优化选项
                options = {
                    "maxiter": 100,
                    "maxfun": 10000,
                    "disp": True,
                    "gtol": 1e-5,
                    "ftol": 1e-8
                }
                
                # 2次优化
                for _ in range(2):
                    result = minimize(
                        fun=one_click_smp_fidelity,
                        x0=parameters,
                        args=(U_theo, local_vars),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=options
                    )
                    parameters = result.x
                    fval = result.fun
                
                # 负相位调整
                for nn in range(t_len, len(parameters)):
                    if parameters[nn] < 0:
                        parameters[nn] += 2 * np.pi

                t = parameters[:t_len]
                theta = parameters[t_len:]
            # 存储1号脉冲结果
            pulse_results[1] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            # 保存Q1_X90参数（供后续衍生脉冲使用）
            local_vars.t_Q1_X90 = t
            local_vars.theta_Q1_X90 = theta
            local_vars.fval = fval
        
        if pulsenumber == 2:
            Pulse_Position = np.array([1, 1, 1, 1, 1]).conj().T
            # 时间复用1号，相位+π/2（Y90°=X90°相位偏移90°）
            t = local_vars.t_Q1_X90
            theta = local_vars.theta_Q1_X90 + np.pi/2
            amp = np.ones(len(t))
            parameters = np.concatenate([t, theta])
            fval = local_vars.fval
            
            # 存储结果
            pulse_results[2] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q1_Y90 = t
            local_vars.theta_Q1_Y90 = theta
        
        if pulsenumber == 3:
            Pulse_Position = np.array([1, 1, 1, 1, 1]).conj().T
            # 时间复用1号，相位+π（反向X90°）
            t = local_vars.t_Q1_X90
            theta = local_vars.theta_Q1_X90 + np.pi
            amp = np.ones(len(t))
            parameters = np.concatenate([t, theta])
            fval = local_vars.fval
            
            pulse_results[3] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q1_X90N = t
            local_vars.theta_Q1_X90N = theta
        
        if pulsenumber == 4:
            Pulse_Position = np.array([1, 1, 1, 1, 1]).conj().T
            t = local_vars.t_Q1_X90
            theta = local_vars.theta_Q1_X90 + 3*np.pi/2
            amp = np.ones(len(t))
            parameters = np.concatenate([t, theta])
            fval = local_vars.fval
            pulse_results[4] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q1_Y90N = t
            local_vars.theta_Q1_Y90N = theta
        
        if pulsenumber == 5:
            Pulse_Position = np.array([2, 2, 2, 2, 2]).conj().T
            U_theo = local_vars.U_theo[4]
            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_Q2_X90 = np.array([6.263000e+01, 8.975000e+01, 1.985800e+02, 6.294000e+01, 3.120000e+00]) * 1e-6
                local_vars.theta_Q2_X90 = np.array([7.548920e+01, 1.171574e+02, 2.968871e+02, 7.607190e+01, 3.601379e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_Q2_X90 = np.array([6.263000e+01, 8.975000e+01, 1.985800e+02, 6.294000e+01, 3.120000e+00]) * 1e-6
                local_vars.theta_Q2_X90 = np.array([7.548920e+01, 1.171574e+02, 2.968871e+02, 7.607190e+01, 3.601379e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_Q2_X90 = np.array([])
                local_vars.theta_Q2_X90 = np.array([])
            
            # 无预定义参数：优化
            if len(local_vars.t_Q2_X90) == 0:
                # PW_H==PW_P：复用Q1_X90参数
                if PW_H == PW_P:
                    local_vars.t_Q2_X90 = local_vars.t_Q1_X90
                    local_vars.theta_Q2_X90 = local_vars.theta_Q1_X90
                    amp = np.ones(len(local_vars.t_Q2_X90))
                    parameters = np.concatenate([local_vars.t_Q2_X90, local_vars.theta_Q2_X90])
                else:
                    # 随机初始化并优化（2自旋系统）
                    local_vars.optim_type = 1
                    local_vars.H_int = H_int_2
                    local_vars.H_pulseHx = H_pulseHx_2
                    local_vars.H_pulseHy = H_pulseHy_2
                    local_vars.H_pulsePx = H_pulsePx_2
                    local_vars.H_pulsePy = H_pulsePy_2
                    fval = 0
                    
                    while -fval / 1e7 <= threshold:
                        # 随机初始化（基于PW_P）
                        t = np.random.rand(len(Pulse_Position)) * 2 * PW_P
                        theta = np.random.rand(len(Pulse_Position)) * 2 * np.pi
                        amp = np.ones(len(t))
                        parameters = np.concatenate([t, theta])
                        t_len = len(t)
                        
                        # 边界约束
                        lb_t = np.full(t_len, -0.0000001e-6)
                        lb_theta = np.full(t_len, -100)
                        lb = np.concatenate([lb_t, lb_theta])
                        
                        ub_t = np.full(t_len, 1000e-6)
                        ub_theta = np.full(t_len, 100)
                        ub = np.concatenate([ub_t, ub_theta])
                        bounds = list(zip(lb, ub))
                        
                        # 优化选项
                        options = {
                            "maxiter": 200,
                            "maxfun": 10000,
                            "disp": False
                        }
                        
                        # 2次优化
                        for _ in range(2):
                            result = minimize(
                                fun=one_click_smp_fidelity,
                                x0=parameters,
                                args=(U_theo, local_vars),
                                method="L-BFGS-B",
                                bounds=bounds,
                                options=options
                            )
                            parameters = result.x
                            fval = result.fun
                        
                        # 负相位调整
                        for nn in range(t_len, len(parameters)):
                            if parameters[nn] < 0:
                                parameters[nn] += 2 * np.pi
                                                
                        t = parameters[:t_len]
                        theta = parameters[t_len:]
                    # 切换到8自旋系统优化
                    U_theo = mykron(I, sqrtX, I, I, I, I, I, I)
                    local_vars.H_int = H_int_6
                    local_vars.H_pulseHx = H_pulseHx_6
                    local_vars.H_pulseHy = H_pulseHy_6
                    local_vars.H_pulsePx = H_pulsePx_6
                    local_vars.H_pulsePy = H_pulsePy_6
                    
                    # 计算理论目标状态
                    local_vars.state_theo = []
                    for kk in range(len(local_vars.rho0_6)):
                        rho_transformed = U_theo @ local_vars.rho0_6[kk] @ U_theo.conj().T
                        rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2,2,2,2,2,2,2,2])
                        local_vars.state_theo.append(rho_traced)
                    
                    # 8自旋优化（2次）
                    local_vars.optim_type = 2
                    options["maxiter"] = 100
                    for _ in range(2):
                        result = minimize(
                            fun=one_click_smp_fidelity,
                            x0=parameters,
                            args=(U_theo, local_vars),
                            method="L-BFGS-B",
                            bounds=bounds,
                            options=options
                        )
                        parameters = result.x
                        fval = result.fun
                    
                    # 负相位调整
                    for nn in range(t_len, len(parameters)):
                        if parameters[nn] < 0:
                            parameters[nn] += 2 * np.pi
                    
                    t = parameters[:t_len]
                    theta = parameters[t_len:]
            
            # 有预定义参数：直接优化
            else:
                local_vars.optim_type = 2
                t = local_vars.t_Q2_X90
                theta = local_vars.theta_Q2_X90
                amp = np.ones(len(t))
                parameters = np.concatenate([t, theta])
                t_len = len(t)
                
                # 8自旋理论幺正变换
                U_theo = mykron(I, sqrtX, I, I, I, I, I, I)
                local_vars.H_int = H_int_6
                local_vars.H_pulseHx = H_pulseHx_6
                local_vars.H_pulseHy = H_pulseHy_6
                local_vars.H_pulsePx = H_pulsePx_6
                local_vars.H_pulsePy = H_pulsePy_6
                
                # 计算理论目标状态
                local_vars.state_theo = []
                for kk in range(len(local_vars.rho0_6)):
                    rho_transformed = U_theo @ local_vars.rho0_6[kk] @ U_theo.conj().T
                    rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2,2,2,2,2,2,2,2])
                    local_vars.state_theo.append(rho_traced)
                
                # 边界约束
                lb_t = np.full(t_len, -0.0000001e-6)
                lb_theta = np.full(t_len, -100)
                lb = np.concatenate([lb_t, lb_theta])
                
                ub_t = np.full(t_len, 1000e-6)
                ub_theta = np.full(t_len, 100)
                ub = np.concatenate([ub_t, ub_theta])
                bounds = list(zip(lb, ub))
                
                # 优化选项
                options = {
                    "maxiter": 100,
                    "maxfun": 10000,
                    "disp": False
                }
                
                # 2次优化
                for _ in range(2):
                    result = minimize(
                        fun=one_click_smp_fidelity,
                        x0=parameters,
                        args=(U_theo, local_vars),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=options
                    )
                    parameters = result.x
                    fval = result.fun
                
                # 负相位调整
                for nn in range(t_len, len(parameters)):
                    if parameters[nn] < 0:
                        parameters[nn] += 2 * np.pi
            
                t = parameters[:t_len]
                theta = parameters[t_len:]
            # 存储5号脉冲结果
            pulse_results[5] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q2_X90 = t
            local_vars.theta_Q2_X90 = theta
            local_vars.fval = fval
        
        if pulsenumber == 6:
            Pulse_Position = np.array([2, 2, 2, 2, 2]).conj().T
            t = local_vars.t_Q2_X90
            theta = local_vars.theta_Q2_X90 + np.pi/2
            amp = np.ones(len(t))
            parameters = np.concatenate([t, theta])
            fval = local_vars.fval
            pulse_results[6] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q2_Y90 = t
            local_vars.theta_Q2_Y90 = theta
        
        if pulsenumber == 7:
            Pulse_Position = np.array([2, 2, 2, 2, 2]).conj().T
            t = local_vars.t_Q2_X90
            theta = local_vars.theta_Q2_X90 + np.pi
            amp = np.ones(len(t))
            parameters = np.concatenate([t, theta])
            fval = local_vars.fval
            pulse_results[7] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q2_X90N = t
            local_vars.theta_Q2_X90N = theta
        
        if pulsenumber == 8:
            Pulse_Position = np.array([2, 2, 2, 2, 2]).conj().T
            t = local_vars.t_Q2_X90
            theta = local_vars.theta_Q2_X90 + 3*np.pi/2
            amp = np.ones(len(t))
            fval = local_vars.fval
            parameters = np.concatenate([t, theta])
            pulse_results[8] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q2_Y90N = t
            local_vars.theta_Q2_Y90N = theta
        
        if pulsenumber == 9:
            Pulse_Position = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).conj().T
            U_theo = mykron(X, I, I, I, I, I, I, I)  # 8自旋X180°幺正变换
            amp = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            # 加载预定义t_Q1_X180
            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_Q1_X180 = np.array([1.017400e+02, 9.719000e+01, 1.985500e+02, 1.015100e+02, 0.000000e+00]) * 1e-6
                local_vars.theta_Q1_X180 = np.array([6.394510e+01, 1.283176e+02, 3.155013e+02, 6.396540e+01, 0.000000e+00]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_Q1_X180 = np.array([1.017400e+02, 9.719000e+01, 1.985500e+02, 1.015100e+02, 0.000000e+00]) * 1e-6
                local_vars.theta_Q1_X180 = np.array([6.394510e+01, 1.283176e+02, 3.155013e+02, 6.396540e+01, 0.000000e+00]) / 360 * 2 * np.pi
            elif (PW_H == 30e-6) and (PW_P == 30e-6):
                local_vars.t_Q1_X180 = np.array([1.275700e+02, 5.379000e+01, 6.395000e+01, 8.810000e+00, 5.229000e+01]) * 1e-6
                local_vars.theta_Q1_X180 = np.array([1.719310e+01, 1.185941e+02, 4.054450e+01, 5.167480e+01, 1.064004e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_Q1_X180 = local_vars.t_Q1_X90  # 复用Q1_X90时间
                local_vars.theta_Q1_X180 = local_vars.theta_Q1_X90  # 复用Q1_X90相位
        
            # 优化参数初始化
            parameters = np.concatenate([local_vars.t_Q1_X180, local_vars.theta_Q1_X180])
            t_len = len(local_vars.t_Q1_X180)
            
            # 哈密顿量（8自旋系统）
            local_vars.H_int = H_int_6
            local_vars.H_pulseHx = H_pulseHx_6
            local_vars.H_pulseHy = H_pulseHy_6
            local_vars.H_pulsePx = H_pulsePx_6
            local_vars.H_pulsePy = H_pulsePy_6
            
            # 计算理论目标状态
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_6)):
                rho_transformed = U_theo @ local_vars.rho0_6[kk] @ U_theo.conj().T
                rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2,2,2,2,2,2,2,2])
                local_vars.state_theo.append(rho_traced)
            
            local_vars.optim_type = 2
            # 边界约束
            lb_t = np.full(t_len, -0.0000001e-6)
            lb_theta = np.full(t_len, -100)
            lb = np.concatenate([lb_t, lb_theta])
            
            ub_t = np.full(t_len, 1000e-6)
            ub_theta = np.full(t_len, 100)
            ub = np.concatenate([ub_t, ub_theta])
            bounds = list(zip(lb, ub))
            
            # 优化选项
            options = {
                "maxiter": 100,
                "maxfun": 10000,
                "disp": False
            }
            
            # 3次优化（原MATLAB for optimize=1:3）
            for _ in range(3):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun
            
            # 负相位调整
            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi
            
            # 更新最终参数
            t = parameters[:t_len]
            theta = parameters[t_len:]
            
            # 存储9号脉冲结果
            pulse_results[9] = {
                "outputfile": outputfile,
                "t": t,
                "theta": theta,
                "amp": amp
            }
            local_vars.t_Q1_X180 = t
            local_vars.theta_Q1_X180 = theta
            local_vars.fval = fval
        
        if pulsenumber == 10:
            Pulse_Position = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).conj().T
            local_vars.t_Q1_Y180 = local_vars.t_Q1_X180
            local_vars.theta_Q1_Y180 = local_vars.theta_Q1_X180 + np.pi/2
            amp = np.ones(len(local_vars.t_Q1_Y180))
            parameters = np.concatenate([local_vars.t_Q1_Y180, local_vars.theta_Q1_Y180])
            fval = local_vars.fval
            pulse_results[10] = {
                "outputfile": outputfile,
                "t": local_vars.t_Q1_Y180,
                "theta": local_vars.theta_Q1_Y180,
                "amp": amp
            }
     
        if pulsenumber == 11:
            Pulse_Position = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).conj().T
            U_theo = mykron(I, X, I, I, I, I, I, I)  # 8自旋Q2_X180幺正变换
            # PW_H==PW_P：复用Q1_X180参数
            if PW_H == PW_P:
                local_vars.t_Q2_X180 = local_vars.t_Q1_X180
                local_vars.theta_Q2_X180 = local_vars.theta_Q1_X180
                parameters = np.concatenate([local_vars.t_Q2_X180, local_vars.theta_Q2_X180])
                amp = np.ones(len(local_vars.t_Q2_X180))
                t_len = len(local_vars.t_Q2_X180) 
                fval = local_vars.fval 
                t = local_vars.t_Q2_X180  # 从local_vars获取t的值
                theta = local_vars.theta_Q2_X180  # 从local_vars获取theta的值
            else:
                # 复用Q2_X90参数（2次拼接）
                local_vars.t_Q2_X180 = np.concatenate([local_vars.t_Q2_X90, local_vars.t_Q2_X90])
                local_vars.theta_Q2_X180 = np.concatenate([local_vars.theta_Q2_X90, local_vars.theta_Q2_X90])
                parameters = np.concatenate([local_vars.t_Q2_X180, local_vars.theta_Q2_X180])
            
                t_len = len(local_vars.t_Q2_X180) 
                # 哈密顿量（8自旋系统）
                local_vars.H_int = H_int_6
                local_vars.H_pulseHx = H_pulseHx_6
                local_vars.H_pulseHy = H_pulseHy_6
                local_vars.H_pulsePx = H_pulsePx_6
                local_vars.H_pulsePy = H_pulsePy_6
                # 计算理论目标状态
                local_vars.state_theo = []
                for kk in range(len(local_vars.rho0_6)):
                    rho_transformed = U_theo @ local_vars.rho0_6[kk] @ U_theo.conj().T
                    rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2,2,2,2,2,2,2,2])
                    local_vars.state_theo.append(rho_traced)
                    
                local_vars.optim_type = 2
                amp = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
                # 边界约束
                lb_t = np.full(t_len, -0.0000001e-6)
                lb_theta = np.full(t_len, -100)
                lb = np.concatenate([lb_t, lb_theta])
                
                ub_t = np.full(t_len, 1000e-6)
                ub_theta = np.full(t_len, 100)
                ub = np.concatenate([ub_t, ub_theta])
                bounds = list(zip(lb, ub))
            
                # 优化选项
                options = {
                    "maxiter": 100,
                    "maxfun": 10000,
                    "disp": False
                }
            
                # 3次优化
                for _ in range(3):
                    result = minimize(
                        fun=one_click_smp_fidelity,
                        x0=parameters,
                        args=(U_theo, local_vars),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=options
                    )
                    parameters = result.x
                    fval = result.fun
            
                # 负相位调整
                for nn in range(t_len, len(parameters)):
                    if parameters[nn] < 0:
                        parameters[nn] += 2 * np.pi
                
                # 更新最终参数
                t = parameters[:t_len]
                theta = parameters[t_len:]
            
            # 存储11号脉冲结果
            pulse_results[11] = {
                "outputfile": outputfile,
                "t": local_vars.t_Q2_X180,
                "theta": local_vars.theta_Q2_X180,
                "amp": amp
            }
            local_vars.t_Q2_X180 = t
            local_vars.theta_Q2_X180 = theta
            local_vars.fval = fval
        
        if pulsenumber == 12:
            Pulse_Position = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]).conj().T
            local_vars.t_Q2_Y180 = local_vars.t_Q2_X180
            local_vars.theta_Q2_Y180 = local_vars.theta_Q2_X180 + np.pi/2
            amp = np.ones(len(local_vars.t_Q2_Y180))
            parameters = np.concatenate([local_vars.t_Q2_Y180, local_vars.theta_Q2_Y180])
            fval = local_vars.fval
            pulse_results[12] = {
                "outputfile": outputfile,
                "t": local_vars.t_Q2_Y180,
                "theta": local_vars.theta_Q2_Y180,
                "amp": amp
            }

        if pulsenumber == 13:
            Pulse_Position = np.array(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1).conj().T 
            U_theo = mykron(Hadamard, I, I, I, I, I, I, I)
            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_Q1_H = np.array([7.840000e+00, 1.673700e+02, 3.966000e+01, 0.000000e+00, 0.000000e+00,
                                            4.008000e+01, 1.585300e+02, 4.057000e+01, 2.264000e+01, 2.000000e-02]) * 1e-6
                local_vars.theta_Q1_H = np.array([4.074548e+02, 3.063382e+02, 4.029028e+02, 2.435900e+02, 3.090000e+02,
                                                3.343166e+02, 2.120802e+02, 3.238698e+02, 1.514154e+02, 2.225500e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_Q1_H = np.array([7.840000e+00, 1.673700e+02, 3.966000e+01, 0.000000e+00, 0.000000e+00,
                                            4.008000e+01, 1.585300e+02, 4.057000e+01, 2.264000e+01, 2.000000e-02]) * 1e-6
                local_vars.theta_Q1_H = np.array([4.074548e+02, 3.063382e+02, 4.029028e+02, 2.435900e+02, 3.090000e+02,
                                                3.343166e+02, 2.120802e+02, 3.238698e+02, 1.514154e+02, 2.225500e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_Q1_H = np.concatenate([local_vars.t_Q1_Y90, local_vars.t_Q1_X180])
                local_vars.theta_Q1_H = np.concatenate([local_vars.theta_Q1_Y90, local_vars.theta_Q1_X180])

            amp = np.ones(len(local_vars.t_Q1_H))
            parameters = np.concatenate([local_vars.t_Q1_H, local_vars.theta_Q1_H])
            t_len = len(local_vars.t_Q1_H)

            local_vars.H_int = H_int_6
            local_vars.H_pulseHx = H_pulseHx_6
            local_vars.H_pulseHy = H_pulseHy_6
            local_vars.H_pulsePx = H_pulsePx_6
            local_vars.H_pulsePy = H_pulsePy_6

            rho_transformed = np.empty((256, 256))
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_6)):
                np.matmul(U_theo, local_vars.rho0_6[kk], out=rho_transformed)  
                np.matmul(rho_transformed, U_theo.conj().T, out=rho_transformed)  
                # 部分迹后转为 complex128，确保 sqrtm 精度
                rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2]*8)
                local_vars.state_theo.append(rho_traced)

            del rho_transformed

            local_vars.optim_type = 2
            MaxIter = 100
            lb_t = np.full(t_len, -0.0000001e-6)
            lb_theta = np.full(t_len, -100)
            lb = np.concatenate([lb_t, lb_theta])
            ub_t = np.full(t_len, 1000e-6)
            ub_theta = np.full(t_len, 100)
            ub = np.concatenate([ub_t, ub_theta])
            bounds = list(zip(lb, ub))

            options = {
                "maxiter": MaxIter,
                "maxfun": 10000,
                "disp": True,  # 打印迭代信息（与MATLAB display='iter'一致）
                "gtol": 1e-5,
                "ftol": 1e-8
            }

            for _ in range(2):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun

            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi

            local_vars.t_Q1_H = parameters[:t_len]
            local_vars.theta_Q1_H = parameters[t_len:]

            pulse_results[13] = {
                "outputfile": outputfile,
                "t": local_vars.t_Q1_H,
                "theta": local_vars.theta_Q1_H,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }

        if pulsenumber == 14:
            Pulse_Position = np.array([2]*15).conj().T  # 15个2的列向量
            U_theo = mykron(I, Hadamard, I, I, I, I, I, I)
            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_Q2_H = np.array([1.020000e+01, 1.691100e+02, 3.802000e+01, 0.000000e+00, 0.000000e+00,
                                            4.116000e+01, 1.584900e+02, 4.016000e+01, 2.231000e+01, 0.000000e+00]) * 1e-6
                local_vars.theta_Q2_H = np.array([4.077969e+02, 3.076776e+02, 4.038369e+02, 2.435900e+02, 3.090000e+02,
                                                3.336016e+02, 2.096091e+02, 3.232101e+02, 1.517592e+02, 2.225500e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_Q2_H = np.array([1.020000e+01, 1.691100e+02, 3.802000e+01, 0.000000e+00, 0.000000e+00,
                                            4.116000e+01, 1.584900e+02, 4.016000e+01, 2.231000e+01, 0.000000e+00]) * 1e-6
                local_vars.theta_Q2_H = np.array([4.077969e+02, 3.076776e+02, 4.038369e+02, 2.435900e+02, 3.090000e+02,
                                                3.336016e+02, 2.096091e+02, 3.232101e+02, 1.517592e+02, 2.225500e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_Q2_H = np.concatenate([local_vars.t_Q2_Y90, local_vars.t_Q2_X180])
                local_vars.theta_Q2_H = np.concatenate([local_vars.theta_Q2_Y90, local_vars.theta_Q2_X180])

            amp = np.ones(len(local_vars.t_Q2_H))
            parameters = np.concatenate([local_vars.t_Q2_H, local_vars.theta_Q2_H])
            t_len = len(local_vars.t_Q2_H)

            local_vars.H_int = H_int_6
            local_vars.H_pulseHx = H_pulseHx_6
            local_vars.H_pulseHy = H_pulseHy_6
            local_vars.H_pulsePx = H_pulsePx_6
            local_vars.H_pulsePy = H_pulsePy_6

            rho_transformed = np.empty((256, 256), dtype=np.complex128)
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_6)):
                np.matmul(U_theo, local_vars.rho0_6[kk], out=rho_transformed)  # complex64
                np.matmul(rho_transformed, U_theo.conj().T, out=rho_transformed)  # complex64
                # 部分迹后转为 complex128，确保 sqrtm 精度
                rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2]*8).astype(np.complex128)
                local_vars.state_theo.append(rho_traced)

            del rho_transformed

            local_vars.optim_type = 2
            MaxIter = 100

            lb_t = np.full(t_len, -0.0000001e-6)
            lb_theta = np.full(t_len, -100)
            lb = np.concatenate([lb_t, lb_theta])
            ub_t = np.full(t_len, 1000e-6)
            ub_theta = np.full(t_len, 100)
            ub = np.concatenate([ub_t, ub_theta])
            bounds = list(zip(lb, ub))

            options = {
                "maxiter": MaxIter,
                "maxfun": 10000,
                "disp": True,
                "gtol": 1e-5,
                "ftol": 1e-8
            }

            for _ in range(3):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun

            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi

            local_vars.t_Q2_H = parameters[:t_len]
            local_vars.theta_Q2_H = parameters[t_len:]

            # 9. 存储结果
            pulse_results[14] = {
                "outputfile": outputfile,
                "t": local_vars.t_Q2_H,
                "theta": local_vars.theta_Q2_H,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }
            
        if pulsenumber == 15:
            Pulse_Position = np.array([2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1]).conj().T
            U_theo = mykron(local_vars.U_theo[14], I, I, I, I, I, I).astype(np.complex128) 

            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_CNOT12 = np.array([2.833000e+01, 1.046300e+02, 1.705600e+02, 3.382000e+01, 1.048000e+01,
                                            7.222200e+02, 2.450000e+01, 3.145000e+01, 1.156300e+02, 4.312000e+01,
                                            2.161000e+01, 5.951000e+01, 3.049000e+01, 1.666300e+02, 4.120000e+01,
                                            0.000000e+00, 8.270000e+00, 4.681000e+01, 1.506300e+02, 3.826000e+01,
                                            2.492000e+01, 6.737000e+01, 3.603000e+01, 1.020600e+02, 3.000000e-02,
                                            4.916000e+01, 2.006000e+01, 1.818000e+01, 1.181300e+02, 3.209000e+01,
                                            5.380000e+00]) * 1e-6
                local_vars.theta_CNOT12 = np.array([1.714226e+02, 2.081438e+02, 3.788251e+02, 1.646909e+02, 4.510432e+02,
                                                3.600000e+02, 2.520837e+02, 3.036913e+02, 4.861605e+02, 2.461388e+02,
                                                5.410789e+02, 3.474521e+02, 3.955087e+02, 5.675207e+02, 3.434796e+02,
                                                6.312265e+02, 1.620182e+02, 2.124191e+02, 3.977226e+02, 1.548285e+02,
                                                4.508166e+02, 7.139830e+01, 1.346765e+02, 2.872915e+02, 7.114400e+01,
                                                3.585140e+02, 7.157040e+01, 1.229695e+02, 3.010679e+02, 6.760170e+01,
                                                3.609772e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_CNOT12 = np.array([2.833000e+01, 1.046300e+02, 1.705600e+02, 3.382000e+01, 1.048000e+01,
                                            7.222200e+02, 2.450000e+01, 3.145000e+01, 1.156300e+02, 4.312000e+01,
                                            2.161000e+01, 5.951000e+01, 3.049000e+01, 1.666300e+02, 4.120000e+01,
                                            0.000000e+00, 8.270000e+00, 4.681000e+01, 1.506300e+02, 3.826000e+01,
                                            2.492000e+01, 6.737000e+01, 3.603000e+01, 1.020600e+02, 3.000000e-02,
                                            4.916000e+01, 2.006000e+01, 1.818000e+01, 1.181300e+02, 3.209000e+01,
                                            5.380000e+00]) * 1e-6
                local_vars.theta_CNOT12 = np.array([1.714226e+02, 2.081438e+02, 3.788251e+02, 1.646909e+02, 4.510432e+02,
                                                3.600000e+02, 2.520837e+02, 3.036913e+02, 4.861605e+02, 2.461388e+02,
                                                5.410789e+02, 3.474521e+02, 3.955087e+02, 5.675207e+02, 3.434796e+02,
                                                6.312265e+02, 1.620182e+02, 2.124191e+02, 3.977226e+02, 1.548285e+02,
                                                4.508166e+02, 7.139830e+01, 1.346765e+02, 2.872915e+02, 7.114400e+01,
                                                3.585140e+02, 7.157040e+01, 1.229695e+02, 3.010679e+02, 6.760170e+01,
                                                3.609772e+02]) / 360 * 2 * np.pi
            elif (PW_H == 30e-6) and (PW_P == 30e-6):
                local_vars.t_CNOT12 = np.array([9.418000e+01, 5.113000e+01, 7.089000e+01, 6.300000e-01, 4.487000e+01,
                                            6.949300e+02, 1.018500e+02, 3.699000e+01, 6.923000e+01, 3.870000e+00,
                                            3.902000e+01, 9.173000e+01, 5.058000e+01, 7.670000e+01, 2.740000e+00,
                                            3.915000e+01, 9.650000e+01, 4.164000e+01, 6.990000e+01, 6.710000e+00,
                                            3.608000e+01, 9.576000e+01, 5.161000e+01, 7.129000e+01, 6.190000e+00,
                                            3.385000e+01, 9.278000e+01, 4.160000e+01, 6.704000e+01, 7.000000e-01,
                                            4.471000e+01]) * 1e-6
                local_vars.theta_CNOT12 = np.array([1.065787e+02, 2.115671e+02, 1.273525e+02, 1.427562e+02, 1.984627e+02,
                                                0.000000e+00, 1.965752e+02, 3.015802e+02, 2.172638e+02, 2.326256e+02,
                                                2.885277e+02, 2.864912e+02, 3.916009e+02, 3.072863e+02, 3.226718e+02,
                                                3.785446e+02, 1.067031e+02, 2.115308e+02, 1.272728e+02, 1.426130e+02,
                                                1.985385e+02, 1.670630e+01, 1.216461e+02, 3.744730e+01, 5.246540e+01,
                                                1.085432e+02, 1.660520e+01, 1.216117e+02, 3.745320e+01, 5.253190e+01,
                                                1.085447e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_CNOT12 = np.concatenate([
                    local_vars.t_Q2_Y90,
                    np.array([718.39e-6]),  # 单独的时间参数
                    local_vars.t_Q1_X90N,
                    local_vars.t_Q2_Y90N,
                    local_vars.t_Q1_Y90,
                    local_vars.t_Q2_X90,
                    local_vars.t_Q1_X90
                ])
                local_vars.theta_CNOT12 = np.concatenate([
                    local_vars.theta_Q2_Y90,
                    np.array([0.0]),  # 对应718.39e-6的相位
                    local_vars.theta_Q1_X90N,
                    local_vars.theta_Q2_Y90N,
                    local_vars.theta_Q1_Y90,
                    local_vars.theta_Q2_X90,
                    local_vars.theta_Q1_X90
                ])

            amp = np.ones(len(local_vars.t_CNOT12))
            amp[5] = 0  # 第6个位置振幅设为0
            parameters = np.concatenate([local_vars.t_CNOT12, local_vars.theta_CNOT12])
            t_len = len(local_vars.t_CNOT12)

            local_vars.H_int = H_int_6
            local_vars.H_pulseHx = H_pulseHx_6
            local_vars.H_pulseHy = H_pulseHy_6
            local_vars.H_pulsePx = H_pulsePx_6
            local_vars.H_pulsePy = H_pulsePy_6

            rho_transformed = np.empty((256, 256), dtype=np.complex128)
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_6)):
                np.matmul(U_theo, local_vars.rho0_6[kk], out=rho_transformed)  # complex64
                np.matmul(rho_transformed, U_theo.conj().T, out=rho_transformed)  # complex64
                # 部分迹后转为 complex128，确保 sqrtm 精度
                rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2]*8).astype(np.complex128)
                local_vars.state_theo.append(rho_traced)

            del rho_transformed


            local_vars.optim_type = 2
            MaxIter = 100

            lb_t = np.full(t_len, -0.0000001e-6)
            lb_theta = np.full(t_len, -100)
            lb = np.concatenate([lb_t, lb_theta])
            ub_t = np.full(t_len, 1000e-6)
            ub_theta = np.full(t_len, 100)
            ub = np.concatenate([ub_t, ub_theta])
            bounds = list(zip(lb, ub))

            options = {
                "maxiter": MaxIter,
                "maxfun": 10000,
                "disp": True,
                "gtol": 1e-5,
                "ftol": 1e-8
            }

            for _ in range(2):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun

            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi

            local_vars.t_CNOT12 = parameters[:t_len]
            local_vars.theta_CNOT12 = parameters[t_len:]

            pulse_results[15] = {
                "outputfile": outputfile,
                "t": local_vars.t_CNOT12,
                "theta": local_vars.theta_CNOT12,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }
 
        if pulsenumber == 16:
            Pulse_Position = np.array([1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2]).conj().T
            U_theo = mykron(local_vars.U_theo[15], I, I, I, I, I, I)

            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_CNOT21 = np.array([2.927000e+01, 1.029500e+02, 1.575000e+02, 3.405000e+01, 1.214000e+01,
                                            6.780500e+02, 2.860000e+00, 1.288000e+01, 6.911000e+01, 4.264000e+01,
                                            2.046000e+01, 5.016000e+01, 3.187000e+01, 1.668500e+02, 4.298000e+01,
                                            5.380000e+00, 7.710000e+00, 2.453000e+01, 1.183800e+02, 2.480000e+01,
                                            4.096000e+01, 6.073000e+01, 2.128000e+01, 7.029000e+01, 3.900000e-01,
                                            6.566000e+01, 1.203000e+01, 8.250000e+00, 9.951000e+01, 2.514000e+01,
                                            6.800000e+00]) * 1e-6
                local_vars.theta_CNOT21 = np.array([1.707028e+02, 2.055310e+02, 3.855429e+02, 1.633300e+02, 4.510939e+02,
                                                3.600000e+02, 2.518035e+02, 3.002748e+02, 4.927362e+02, 2.441639e+02,
                                                5.400339e+02, 3.398168e+02, 4.009330e+02, 5.695790e+02, 3.440070e+02,
                                                6.311056e+02, 1.616874e+02, 2.090773e+02, 4.051245e+02, 1.533836e+02,
                                                4.503079e+02, 6.568390e+01, 1.362463e+02, 2.937765e+02, 7.114240e+01,
                                                3.495318e+02, 7.071730e+01, 1.222158e+02, 3.018823e+02, 6.644970e+01,
                                                3.610216e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_CNOT21 = np.array([2.927000e+01, 1.029500e+02, 1.575000e+02, 3.405000e+01, 1.214000e+01,
                                            6.780500e+02, 2.860000e+00, 1.288000e+01, 6.911000e+01, 4.264000e+01,
                                            2.046000e+01, 5.016000e+01, 3.187000e+01, 1.668500e+02, 4.298000e+01,
                                            5.380000e+00, 7.710000e+00, 2.453000e+01, 1.183800e+02, 2.480000e+01,
                                            4.096000e+01, 6.073000e+01, 2.128000e+01, 7.029000e+01, 3.900000e-01,
                                            6.566000e+01, 1.203000e+01, 8.250000e+00, 9.951000e+01, 2.514000e+01,
                                            6.800000e+00]) * 1e-6
                local_vars.theta_CNOT21 = np.array([1.707028e+02, 2.055310e+02, 3.855429e+02, 1.633300e+02, 4.510939e+02,
                                                3.600000e+02, 2.518035e+02, 3.002748e+02, 4.927362e+02, 2.441639e+02,
                                                5.400339e+02, 3.398168e+02, 4.009330e+02, 5.695790e+02, 3.440070e+02,
                                                6.311056e+02, 1.616874e+02, 2.090773e+02, 4.051245e+02, 1.533836e+02,
                                                4.503079e+02, 6.568390e+01, 1.362463e+02, 2.937765e+02, 7.114240e+01,
                                                3.495318e+02, 7.071730e+01, 1.222158e+02, 3.018823e+02, 6.644970e+01,
                                                3.610216e+02]) / 360 * 2 * np.pi
            elif (PW_H == 30e-6) and (PW_P == 30e-6):
                local_vars.t_CNOT21 = np.array([9.478000e+01, 5.298000e+01, 6.935000e+01, 1.470000e+00, 4.623000e+01,
                                            7.019800e+02, 8.555000e+01, 4.536000e+01, 7.339000e+01, 5.420000e+00,
                                            3.480000e+01, 9.711000e+01, 5.081000e+01, 8.016000e+01, 5.000000e-02,
                                            2.562000e+01, 8.731000e+01, 5.149000e+01, 6.577000e+01, 1.000000e-02,
                                            3.278000e+01, 1.089100e+02, 5.232000e+01, 6.956000e+01, 1.490000e+00,
                                            3.750000e+01, 8.420000e+01, 4.433000e+01, 6.862000e+01, 5.000000e-02,
                                            3.113000e+01]) * 1e-6
                local_vars.theta_CNOT21 = np.array([1.063866e+02, 2.114659e+02, 1.270646e+02, 1.424225e+02, 1.986483e+02,
                                                3.600000e+02, 1.963283e+02, 3.014671e+02, 2.172524e+02, 2.325215e+02,
                                                2.885055e+02, 2.863388e+02, 3.915902e+02, 3.074262e+02, 3.225642e+02,
                                                3.785672e+02, 1.064890e+02, 2.115310e+02, 1.273743e+02, 1.426524e+02,
                                                1.984217e+02, 1.642170e+01, 1.218556e+02, 3.741540e+01, 5.260410e+01,
                                                1.081263e+02, 1.660610e+01, 1.216927e+02, 3.685070e+01, 5.361830e+01,
                                                1.087586e+02]) / 360 * 2 * np.pi
            else:
                local_vars.t_CNOT21 = np.concatenate([
                    local_vars.t_Q1_Y90,
                    np.array([718.39e-6]),
                    local_vars.t_Q2_X90N,
                    local_vars.t_Q1_Y90N,
                    local_vars.t_Q2_Y90,
                    local_vars.t_Q1_X90,
                    local_vars.t_Q2_X90
                ])
                local_vars.theta_CNOT21 = np.concatenate([
                    local_vars.theta_Q1_Y90,
                    np.array([0.0]),
                    local_vars.theta_Q2_X90N,
                    local_vars.theta_Q1_Y90N,
                    local_vars.theta_Q2_Y90,
                    local_vars.theta_Q1_X90,
                    local_vars.theta_Q2_X90
                ])

            amp = np.ones(len(local_vars.t_CNOT21))
            amp[5] = 0
            parameters = np.concatenate([local_vars.t_CNOT21, local_vars.theta_CNOT21])
            t_len = len(local_vars.t_CNOT21)

            local_vars.H_int = H_int_6
            local_vars.H_pulseHx = H_pulseHx_6
            local_vars.H_pulseHy = H_pulseHy_6
            local_vars.H_pulsePx = H_pulsePx_6
            local_vars.H_pulsePy = H_pulsePy_6

            rho_transformed = np.empty((256, 256), dtype=np.complex128)
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_6)):
                np.matmul(U_theo, local_vars.rho0_6[kk], out=rho_transformed)  # complex64
                np.matmul(rho_transformed, U_theo.conj().T, out=rho_transformed)  # complex64
                # 部分迹后转为 complex128，确保 sqrtm 精度
                rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2]*8).astype(np.complex128)
                local_vars.state_theo.append(rho_traced)

            del rho_transformed


            local_vars.optim_type = 2
            MaxIter = 100

            lb_t = np.full(t_len, -0.0000001e-6)
            lb_theta = np.full(t_len, -100)
            lb = np.concatenate([lb_t, lb_theta])
            ub_t = np.full(t_len, 1000e-6)
            ub_theta = np.full(t_len, 100)
            ub = np.concatenate([ub_t, ub_theta])
            bounds = list(zip(lb, ub))

            options = {
                "maxiter": MaxIter,
                "maxfun": 10000,
                "disp": True,
                "gtol": 1e-5,
                "ftol": 1e-8
            }

            for _ in range(2):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun

            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi

            local_vars.t_CNOT21 = parameters[:t_len]
            local_vars.theta_CNOT21 = parameters[t_len:]

            pulse_results[16] = {
                "outputfile": outputfile,
                "t": local_vars.t_CNOT21,
                "theta": local_vars.theta_CNOT21,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }

        if pulsenumber == 17:
            Pulse_Position = np.array([1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1]).conj().T
            U_theo = mykron(local_vars.U_theo[16], I, I, I, I, I, I)
            if (PW_H == 50e-6) and (PW_P == 50e-6):
                local_vars.t_CZ = np.array([7.798000e+01, 3.340000e+00, 1.305700e+02, 5.606000e+01, 3.835000e+01,
                                        6.299000e+01, 6.698000e+01, 1.427100e+02, 1.100000e-01, 6.904000e+01,
                                        8.853000e+01, 1.340000e+00, 1.310200e+02, 6.000000e+00, 3.023000e+01,
                                        6.001000e+01, 2.688000e+01, 1.811300e+02, 8.384000e+01, 0.000000e+00,
                                        1.037200e+02, 1.034700e+02, 1.655800e+02, 6.000000e-02, 9.855000e+01,
                                        4.900000e+01, 1.466000e+01, 1.394600e+02, 0.000000e+00, 1.300000e-01,
                                        7.037200e+02]) * 1e-6
                local_vars.theta_CZ = np.array([3.412066e+02, 3.883349e+02, 5.704312e+02, 3.469294e+02, 6.295965e+02,
                                            3.489984e+02, 3.897223e+02, 5.620522e+02, 3.485868e+02, 6.304074e+02,
                                            7.238990e+01, 1.167277e+02, 3.023184e+02, 7.581590e+01, 3.608950e+02,
                                            7.336220e+01, 1.171474e+02, 3.001431e+02, 7.858520e+01, 3.601388e+02,
                                            1.665214e+02, 2.044505e+02, 3.867868e+02, 1.657413e+02, 4.532078e+02,
                                            1.668162e+02, 2.063610e+02, 3.837342e+02, 1.684417e+02, 4.501412e+02,
                                            3.600000e+02]) / 360 * 2 * np.pi
            elif (PW_H == 40e-6) and (PW_P == 40e-6):
                local_vars.t_CZ = np.array([7.798000e+01, 3.340000e+00, 1.305700e+02, 5.606000e+01, 3.835000e+01,
                                        6.299000e+01, 6.698000e+01, 1.427100e+02, 1.100000e-01, 6.904000e+01,
                                        8.853000e+01, 1.340000e+00, 1.310200e+02, 6.000000e+00, 3.023000e+01,
                                        6.001000e+01, 2.688000e+01, 1.811300e+02, 8.384000e+01, 0.000000e+00,
                                        1.037200e+02, 1.034700e+02, 1.655800e+02, 6.000000e-02, 9.855000e+01,
                                        4.900000e+01, 1.466000e+01, 1.394600e+02, 0.000000e+00, 1.300000e-01,
                                        7.037200e+02]) * 1e-6
                local_vars.theta_CZ = np.array([3.412066e+02, 3.883349e+02, 5.704312e+02, 3.469294e+02, 6.295965e+02,
                                            3.489984e+02, 3.897223e+02, 5.620522e+02, 3.485868e+02, 6.304074e+02,
                                            7.238990e+01, 1.167277e+02, 3.023184e+02, 7.581590e+01, 3.608950e+02,
                                            7.336220e+01, 1.171474e+02, 3.001431e+02, 7.858520e+01, 3.601388e+02,
                                            1.665214e+02, 2.044505e+02, 3.867868e+02, 1.657413e+02, 4.532078e+02,
                                            1.668162e+02, 2.063610e+02, 3.837342e+02, 1.684417e+02, 4.501412e+02,
                                            3.600000e+02]) / 360 * 2 * np.pi
            elif (PW_H == 30e-6) and (PW_P == 30e-6):
                local_vars.t_CZ = np.array([9.597000e+01, 5.409000e+01, 7.023000e+01, 8.400000e-01, 4.672000e+01,
                                        9.596000e+01, 5.413000e+01, 7.016000e+01, 7.000000e-01, 4.660000e+01,
                                        9.609000e+01, 5.413000e+01, 7.015000e+01, 8.100000e-01, 4.667000e+01,
                                        9.665000e+01, 5.423000e+01, 7.005000e+01, 1.130000e+00, 4.671000e+01,
                                        9.616000e+01, 5.407000e+01, 7.009000e+01, 8.400000e-01, 4.673000e+01,
                                        9.565000e+01, 5.430000e+01, 7.037000e+01, 7.300000e-01, 4.660000e+01,
                                        7.174500e+02]) * 1e-6
                local_vars.theta_CZ = np.array([2.863559e+02, 3.915380e+02, 3.073183e+02, 3.226207e+02, 3.785567e+02,
                                            2.866786e+02, 3.914337e+02, 3.075298e+02, 3.227260e+02, 3.782736e+02,
                                            1.644330e+01, 1.217212e+02, 3.789940e+01, 5.225970e+01, 1.084675e+02,
                                            1.703260e+01, 1.216065e+02, 3.779420e+01, 5.275200e+01, 1.084610e+02,
                                            1.065129e+02, 2.116167e+02, 1.272256e+02, 1.427603e+02, 1.986120e+02,
                                            1.060218e+02, 2.111762e+02, 1.272580e+02, 1.425060e+02, 1.980296e+02,
                                            0.000000e+00]) / 360 * 2 * np.pi
            else:
                local_vars.t_CZ = np.concatenate([
                        local_vars.t_Q1_Y90N,
                        local_vars.t_Q2_Y90N,
                        local_vars.t_Q1_X90,
                        local_vars.t_Q2_X90,
                        local_vars.t_Q1_Y90,
                        local_vars.t_Q2_Y90,
                        np.array([718.39e-6]),
                    ])
                local_vars.theta_CZ = np.concatenate([
                        local_vars.theta_Q1_Y90N,
                        local_vars.theta_Q2_Y90N,
                        local_vars.theta_Q1_X90,
                        local_vars.theta_Q2_X90,
                        local_vars.theta_Q1_Y90,
                        local_vars.theta_Q2_Y90,
                        np.array([0.0]),
                ])

            assert len(local_vars.t_CZ) == 31, f"t_CZ长度应为31，实际为{len(local_vars.t_CZ)}"
            amp = np.ones(31)
            amp[30] = 0  # 正确对应MATLAB的amp(31)=0
            parameters = np.concatenate([local_vars.t_CZ, local_vars.theta_CZ])
            t_len = len(local_vars.t_CZ)

            local_vars.H_int = H_int_6
            local_vars.H_pulseHx = H_pulseHx_6
            local_vars.H_pulseHy = H_pulseHy_6
            local_vars.H_pulsePx = H_pulsePx_6
            local_vars.H_pulsePy = H_pulsePy_6

            rho_transformed = np.empty((256, 256), dtype=np.complex128)
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_6)):
                np.matmul(U_theo, local_vars.rho0_6[kk], out=rho_transformed)  # complex64
                np.matmul(rho_transformed, U_theo.conj().T, out=rho_transformed)  # complex64
                # 部分迹后转为 complex128，确保 sqrtm 精度
                rho_traced = partial_trace(rho_transformed, [2,3,4,5,6,7], [2]*8).astype(np.complex128)
                local_vars.state_theo.append(rho_traced)

            del rho_transformed


            local_vars.optim_type = 2
            MaxIter = 100

            lb_t = np.full(t_len, -0.0000001e-6)
            lb_theta = np.full(t_len, -100)
            lb = np.concatenate([lb_t, lb_theta])
            ub_t = np.full(t_len, 1000e-6)
            ub_theta = np.full(t_len, 100)
            ub = np.concatenate([ub_t, ub_theta])
            bounds = list(zip(lb, ub))

            options = {
                "maxiter": MaxIter,
                "maxfun": 10000,
                "disp": True,
                "gtol": 1e-5,
                "ftol": 1e-8
            }

            for _ in range(2):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun

            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi

            local_vars.t_CZ = parameters[:t_len]
            local_vars.theta_CZ = parameters[t_len:]
            local_vars.fval = fval
            pulse_results[17] = {
                "outputfile": outputfile,
                "t": local_vars.t_CZ,
                "theta": local_vars.theta_CZ,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }

        if pulsenumber == 18:
            Pulse_Position=np.array([2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1]).conj().T
            local_vars.t_SWAP = np.concatenate([local_vars.t_CNOT12, local_vars.t_CNOT21, local_vars.t_CNOT12])
            local_vars.theta_SWAP = np.concatenate([local_vars.theta_CNOT12, local_vars.theta_CNOT21, local_vars.theta_CNOT12])
            amp = np.ones(len(local_vars.t_SWAP))
            parameters = np.concatenate([local_vars.t_SWAP, local_vars.theta_SWAP])
            fval = local_vars.fval
            pulse_results[18] = {
                "outputfile": outputfile,
                "t": local_vars.t_SWAP,
                "theta": local_vars.theta_SWAP,
                "amp": amp
            }
        
        if pulsenumber == 19:
            U_theo = mykron(U_lineselective, sqrtY)
            Pulse_Position = np.array([1,1,1,1,2,2,2,2,1,1,1,1,1,2,2,2,2,1,1,1,1,1,2,2,2,2]).conj().T
            local_vars.H_int = H_int_3
            local_vars.H_pulseHx = H_pulseHx_3
            local_vars.H_pulseHy = H_pulseHy_3
            local_vars.H_pulsePx = H_pulsePx_3
            local_vars.H_pulsePy = H_pulsePy_3
            local_vars.optim_type = 2
            fval = 0
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_PPS1_3)):
                transformed = U_theo @ local_vars.rho0_PPS1_3[kk] @ U_theo.conj().T
                local_vars.state_theo.append(transformed)
            while -fval / 1e7 <= threshold:
                if PW_H == 50e-6 and PW_P == 50e-6:
                    t = np.array([4.287000e+01, 1.050000e+01, 5.637000e+01, 8.400000e-01, 
                                2.143000e+01, 0.000000e+00, 7.118000e+01, 3.131000e+01, 
                                5.227300e+02, 2.104000e+01, 4.070000e+01, 1.568000e+01, 
                                1.018100e+02, 4.058000e+01, 3.700000e-01, 4.600000e-01, 
                                3.040000e+01, 5.009000e+02, 7.139000e+01, 3.165000e+01, 
                                5.476000e+01, 2.647000e+01, 2.010000e+00, 1.354000e+01, 
                                8.236000e+01, 4.254000e+01]) * 1e-6
                    theta = np.array([2.250300e+02, 3.251000e+01, 2.938000e+01, 2.790000e+02, 
                                    3.259200e+02, 1.919200e+02, 3.873000e+01, 2.968300e+02, 
                                    1.217200e+02, 1.050000e+02, 2.679500e+02, 3.710000e+00, 
                                    1.715000e+01, 2.397100e+02, 2.169200e+02, 1.898000e+02, 
                                    2.621600e+02, 2.546100e+02, 2.796700e+02, 1.028400e+02, 
                                    2.493500e+02, 2.004300e+02, 1.425200e+02, 2.193000e+01, 
                                    2.805800e+02, 1.216500e+02]) / 360 * 2 * np.pi               
                elif PW_H == 40e-6 and PW_P == 40e-6:
                    t = np.array([4.287000e+01, 1.050000e+01, 5.637000e+01, 8.400000e-01, 
                                2.143000e+01, 0.000000e+00, 7.118000e+01, 3.131000e+01, 
                                5.227300e+02, 2.104000e+01, 4.070000e+01, 1.568000e+01, 
                                1.018100e+02, 4.058000e+01, 3.700000e-01, 4.600000e-01, 
                                3.040000e+01, 5.009000e+02, 7.139000e+01, 3.165000e+01, 
                                5.476000e+01, 2.647000e+01, 2.010000e+00, 1.354000e+01, 
                                8.236000e+01, 4.254000e+01]) * 1e-6                
                    theta = np.array([2.250300e+02, 3.251000e+01, 2.938000e+01, 2.790000e+02, 
                                    3.259200e+02, 1.919200e+02, 3.873000e+01, 2.968300e+02, 
                                    1.217200e+02, 1.050000e+02, 2.679500e+02, 3.710000e+00, 
                                    1.715000e+01, 2.397100e+02, 2.169200e+02, 1.898000e+02, 
                                    2.621600e+02, 2.546100e+02, 2.796700e+02, 1.028400e+02, 
                                    2.493500e+02, 2.004300e+02, 1.425200e+02, 2.193000e+01, 
                                    2.805800e+02, 1.216500e+02]) / 360 * 2 * np.pi    
                else:
                    t = np.array([4.287000e+01, 1.050000e+01, 5.637000e+01, 8.400000e-01,
                                2.143000e+01, 0.000000e+00, 7.118000e+01, 3.131000e+01,
                                5.227300e+02, 2.104000e+01, 4.070000e+01, 1.568000e+01,
                                1.018100e+02, 4.058000e+01, 3.700000e-01, 4.600000e-01,
                                3.040000e+01, 5.009000e+02, 7.139000e+01, 3.165000e+01,
                                5.476000e+01, 2.647000e+01, 2.010000e+00, 1.354000e+01,
                                8.236000e+01,4.254000e+01]) * 1e-6   
                    theta = np.array([2.250300e+02, 3.251000e+01, 2.938000e+01, 2.790000e+02,
                                    3.259200e+02, 1.919200e+02, 3.873000e+01, 2.968300e+02,
                                    1.217200e+02, 1.050000e+02, 2.679500e+02, 3.710000e+00,
                                    1.715000e+01, 2.397100e+02, 2.169200e+02, 1.898000e+02,
                                    2.621600e+02, 2.546100e+02, 2.796700e+02, 1.028400e+02,
                                    2.493500e+02, 2.004300e+02, 1.425200e+02, 2.193000e+01,
                                    2.805800e+02,1.216500e+02]) / 360 * 2 * np.pi  
                
                amp = np.ones(len(t))
                amp[8] = 0  # 第9个元素
                amp[17] = 0  # 第18个元素
                amp = np.append(amp, 0) 
                
                parameters = np.concatenate([t, theta])
                t_len = len(t)
                
                # 设置边界约束
                lb_t = np.full(t_len, -0.0000001e-6)
                lb_theta = np.full(t_len, -100)
                lb = np.concatenate([lb_t, lb_theta])
                
                ub_t = np.full(t_len, 1000e-6)
                ub_theta = np.full(t_len, 100)
                ub = np.concatenate([ub_t, ub_theta])
                bounds = list(zip(lb, ub))
                
                # 优化选项
                options = {
                    "maxiter": 200,
                    "maxfun": 10000,
                    "disp": True  # 显示迭代信息
                }
                
                # 两次优化
                for _ in range(2):
                    result = minimize(
                        fun=one_click_smp_fidelity,
                        x0=parameters,
                        args=(U_theo, local_vars),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=options
                    )
                    parameters = result.x
                    fval = result.fun
            
            # 调整负相位
            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi
            
            # 提取结果
            local_vars.t_PPS_PART1 = parameters[:t_len]
            local_vars.theta_PPS_PART1 = parameters[t_len:]
            
            # 额外参数设置
            if len(parameters) > 2 * t_len:  # 检查是否有第27个参数
                parameters[t_len] = 3.000000e-1  # t的第27个参数
                parameters[2 * t_len] = 0  # theta的第27个参数
            Pulse_Position = np.append(Pulse_Position, 1)  # 添加第27个位置
            
            # 存储结果
            pulse_results[19] = {
                "outputfile": outputfile,
                "t": local_vars.t_PPS_PART1,
                "theta": local_vars.theta_PPS_PART1,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }

        elif pulsenumber == 20:
            U_theo = local_vars.U_permute
            Pulse_Position = np.array([1,1,1,1,1,1,2,2,2,2,2,1,1,1,1,1,1,2,2,2,2,2]).conj().T
            
            local_vars.H_int = H_int_2
            local_vars.H_pulseHx = H_pulseHx_2
            local_vars.H_pulseHy = H_pulseHy_2
            local_vars.H_pulsePx = H_pulsePx_2
            local_vars.H_pulsePy = H_pulsePy_2
            local_vars.optim_type = 1
            fval = 0
            
            # 优化循环
            while -fval / 1e7 <= threshold:
                if PW_H == 50e-6 and PW_P == 50e-6:
                    t = np.array([8.000000e-02, 0.000000e+00, 8.881000e+01, 1.007000e+02, 
                                3.846000e+01, 7.158800e+02, 6.767000e+01, 3.090000e+01, 
                                1.409800e+02, 1.500000e-01, 1.460000e+00, 7.700000e+01, 
                                0.000000e+00, 1.359300e+02, 7.320000e+01, 2.107000e+01, 
                                5.888500e+02, 6.830000e+01, 5.800000e-01, 6.371000e+01, 
                                0.000000e+00, 3.539000e+01]) * 1e-6
                    
                    theta = np.array([1.596131e+02, 2.084267e+02, 3.969625e+02, 1.610836e+02, 
                                    4.513154e+02, 3.600000e+02, 1.847534e+02, 1.997549e+02, 
                                    3.833306e+02, 1.678308e+02, 4.513385e+02, 8.105700e+01, 
                                    1.182300e+02, 2.939407e+02, 7.336060e+01, 3.632771e+02, 
                                    3.600000e+02, 7.456990e+01, 1.221538e+02, 2.955400e+02, 
                                    7.697970e+01, 3.584556e+02]) / 360 * 2 * np.pi
                
                elif PW_H == 40e-6 and PW_P == 40e-6:
                    t = np.array([8.000000e-02, 0.000000e+00, 8.881000e+01, 1.007000e+02, 
                                3.846000e+01, 7.158800e+02, 6.767000e+01, 3.090000e+01, 
                                1.409800e+02, 1.500000e-01, 1.460000e+00, 7.700000e+01, 
                                0.000000e+00, 1.359300e+02, 7.320000e+01, 2.107000e+01, 
                                5.888500e+02, 6.830000e+01, 5.800000e-01, 6.371000e+01, 
                                0.000000e+00, 3.539000e+01]) * 1e-6
                    
                    theta = np.array([1.596131e+02, 2.084267e+02, 3.969625e+02, 1.610836e+02, 
                                    4.513154e+02, 3.600000e+02, 1.847534e+02, 1.997549e+02, 
                                    3.833306e+02, 1.678308e+02, 4.513385e+02, 8.105700e+01, 
                                    1.182300e+02, 2.939407e+02, 7.336060e+01, 3.632771e+02, 
                                    3.600000e+02, 7.456990e+01, 1.221538e+02, 2.955400e+02, 
                                    7.697970e+01, 3.584556e+02]) / 360 * 2 * np.pi
                
                elif PW_H == 30e-6 and PW_P == 30e-6:
                    t = np.array([2.254000e+01, 6.193000e+01, 1.220000e+00, 7.210000e+01, 
                                6.040000e+00, 6.960100e+02, 1.649000e+01, 3.068000e+01, 
                                0.000000e+00, 0.000000e+00, 8.440000e+00, 1.750000e+00, 
                                3.317000e+01, 5.038000e+01, 6.486000e+01, 1.758000e+01, 
                                6.711600e+02, 0.000000e+00, 5.295000e+01, 1.660000e+00, 
                                3.058000e+01, 2.930000e+01]) * 1e-6
                    
                    theta = np.array([1.042100e+01, 1.623518e+02, 6.901500e+01, 2.455990e+02, 
                                    1.363705e+02, 3.029616e+02, 2.687907e+02, 2.061316e+02, 
                                    6.599590e+01, 3.361501e+02, 1.009797e+02, 3.303430e+02, 
                                    8.033750e+01, 1.488163e+02, 1.380190e+01, 2.434844e+02, 
                                    6.502210e+01, 1.499850e+01, 2.471411e+02, 1.283418e+02, 
                                    2.369312e+02, 1.394770e+02]) / 360 * 2 * np.pi
                
                else:
                    # 随机初始化
                    t_len = len(Pulse_Position)
                    t = np.random.rand(1, t_len) * PW_H
                    t[0, 5] = 718e-6  # 第6个元素（0索引为5）
                    t[0, 16] = 718e-6  # 第17个元素（0索引为16）
                    t = t.flatten()
                    
                    theta = np.random.rand(1, t_len) * 2 * np.pi
                    theta = theta.flatten()
                
                # 设置振幅
                amp = np.ones(len(t))
                amp[5] = 0  # 第6个元素
                amp[16] = 0  # 第17个元素
                
                # 组合参数
                parameters = np.concatenate([t, theta])
                t_len = len(t)
                
                # 设置边界约束
                lb_t = np.full(t_len, -0.0000001e-6)
                lb_theta = np.full(t_len, -100)
                lb = np.concatenate([lb_t, lb_theta])
                
                ub_t = np.full(t_len, 1000e-6)
                ub_theta = np.full(t_len, 100)
                ub = np.concatenate([ub_t, ub_theta])
                bounds = list(zip(lb, ub))
                
                # 优化选项
                options = {
                    "maxiter": 100,
                    "maxfun": 10000,
                    "disp": True
                }
                
                # 两次优化
                for _ in range(2):
                    result = minimize(
                        fun=one_click_smp_fidelity,
                        x0=parameters,
                        args=(U_theo, local_vars),
                        method="L-BFGS-B",
                        bounds=bounds,
                        options=options
                    )
                    parameters = result.x
                    fval = result.fun
            
            # 更新理论幺正变换和哈密顿量
            U_theo = mykron(local_vars.U_permute, I)
            local_vars.H_int = H_int_3
            local_vars.H_pulseHx = H_pulseHx_3
            local_vars.H_pulseHy = H_pulseHy_3
            local_vars.H_pulsePx = H_pulsePx_3
            local_vars.H_pulsePy = H_pulsePy_3
            
            # 计算部分迹后的理论状态
            local_vars.state_theo = []
            for kk in range(len(local_vars.rho0_3)):
                transformed = U_theo @ local_vars.rho0_3[kk] @ U_theo.conj().T
                traced = partial_trace(transformed, [0], [2, 2, 2])  # 保留第一个量子比特
                local_vars.state_theo.append(traced)
            
            # 第二次优化
            local_vars.optim_type = 2
            options["maxiter"] = 100
            
            for _ in range(2):
                result = minimize(
                    fun=one_click_smp_fidelity,
                    x0=parameters,
                    args=(U_theo, local_vars),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options
                )
                parameters = result.x
                fval = result.fun
            
            # 调整负相位
            for nn in range(t_len, len(parameters)):
                if parameters[nn] < 0:
                    parameters[nn] += 2 * np.pi
            
            # 提取结果
            t_PPS_PART2 = parameters[:t_len]
            theta_PPS_PART2 = parameters[t_len:]
            
            # 存储结果
            pulse_results[20] = {
                "outputfile": outputfile,
                "t": t_PPS_PART2,
                "theta": theta_PPS_PART2,
                "amp": amp,
                "Pulse_Position": Pulse_Position
            }
        if File_Type == 1:
            write_pulse_parameters(parameters, amp, Pulse_Position, outputfile, 
                          fval, pulse_width_H, pulse_width_P)
        else:
            write_pulse_json(parameters, amp, Pulse_Position, outputfile, 
                          fval, pulse_width_H, pulse_width_P)
    except Exception as e:
        result_queue.put({'success': False, 'error': str(e)})
        print(f"脉冲优化线程错误: {str(e)}")

if __name__ == "__main__":
    """主函数：设置参数并启动多线程处理"""
    total_start_time = time.time()  # 总计时开始

    P1_methyl = -101.93 # 甲基左侧峰频率(Hz)
    P2_methyl = -90 # 甲基右侧峰频率(Hz)
    w_H = 27.029290 * 10**6 # H通道频率(Hz)
    w_P = 10.941700 * 10**6# P通道频率(Hz)
    PW_H = 40e-6 # H通道90°脉宽(s)
    PW_P = 40e-6 # P通道90°脉宽(s)
    T_indoor = 24 + 273.15  # 室内温度（开尔文）

    w_1 = 0
    w_2 = 0
    # 不同核之间的J耦合频率
    J12 = 696 # 不同核之间的J耦合频率
    w_3 = (P1_methyl + P2_methyl) / 2 
    w_4 = 170
    J23 = abs(P1_methyl - P2_methyl) # 甲基劈裂的两个峰频率差
    o1 = [-30, 0, 30] # 考虑温漂
    o2 = [x * w_P / w_H for x in o1] # mini
    pulse_width_H = np.array([0.95, 1, 1.05]) * PW_H
    pulse_width_P = np.array([0.95, 1, 1.05]) * PW_P
    File_Type = 1

    pulse_params_list = [
        # (1, 'Gemini_Q1_X90.spinq', 0.9996),
        # (2, 'Gemini_Q1_Y90.spinq', 0.9996),
        # (3, 'Gemini_Q1_X90N.spinq', 0.9996),
        # (4, 'Gemini_Q1_Y90N.spinq', 0.9996),
        # (5, 'Gemini_Q2_X90.spinq', 0.9996),
        # (6, 'Gemini_Q2_Y90.spinq', 0.9996),
        # (7, 'Gemini_Q2_X90N.spinq', 0.9996),
        # (8, 'Gemini_Q2_Y90N.spinq', 0.9996),
        # (9, 'Gemini_Q1_X180.spinq', 0.9996),
        # (10, 'Gemini_Q1_Y180.spinq', 0.9996),
        # (11, 'Gemini_Q2_X180.spinq', 0.9996),
        # (12, 'Gemini_Q2_Y180.spinq', 0.9996),
        # (13, 'Gemini_Q1_H.spinq', 0.9996),
        # (14, 'Gemini_Q2_H.spinq', 0.9996),
        # (15, 'Gemini_CNOT12.spinq', 0.9996),
        # (16, 'Gemini_CNOT21.spinq', 0.999),
        # (17, 'Gemini_CZ.spinq', 0.999),
        # (18, 'Gemini_SWAP.spinq', 0.999),
        (19, 'PPS_PART1.spinq', 0.998),
        (20, 'PPS_PART2.spinq', 0.998)
    ]

    fit_params = (
        PW_H, PW_P, T_indoor, w_H, w_P, J12, J23,
        w_1, w_2, w_3, w_4, o1, o2, pulse_width_H, pulse_width_P, File_Type
    )

    pulse_result_queue = queue.Queue()

    for idx, pulse_params in enumerate(pulse_params_list, 1):
        # 显示当前处理进度
        total_pulses = len(pulse_params_list)
        pulsenumber = pulse_params[0]
        outputfile = pulse_params[1]
        local_vars.pulsenumber = pulsenumber
        print(f"\n=== 正在处理第 {idx}/{total_pulses} 个脉冲 ===")
        print(f"脉冲编号: {pulsenumber}, 输出文件: {outputfile}")
        
        # 创建单个线程
        empty_queue = queue.Queue()
        thread = threading.Thread(
            target=optimize_pulse,
            args=(empty_queue, pulse_result_queue, fit_params, pulse_params, local_vars)
        )
        
        # 启动线程并等待完成
        thread.start()
        thread.join()  # 阻塞等待当前脉冲优化完成
        
    # 所有脉冲处理完成后，汇总结果
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"\n所有脉冲优化完成，总耗时: {total_time:.2f}秒")