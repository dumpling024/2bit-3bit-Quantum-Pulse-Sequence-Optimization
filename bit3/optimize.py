import numpy as np
import scipy.linalg as nla
from scipy.linalg import expm
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.interpolate import splrep, splev
import threading
import traceback
import os
from scipy.linalg import sqrtm
import warnings
from .write import write_pulse_file, write_pulse_file_json, dataout_SpinQ
warnings.filterwarnings("ignore", category=DeprecationWarning)
thread_lock = threading.Lock()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

# 全局变量定义
class GlobalVars:
     def __init__(self):
        self.H_int = []
        self.Uideal = None
        self.M = 0              
        self.t = 0.0            
        self.I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        self.X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        self.Ix = self.X / 2
        self.Iy = self.Y / 2
        self.Iz = self.Z / 2
        self.Sigma = 5e+4
        self.RF_deviation = np.array([0.95, 1, 1.05])
        self.Spatial_dist = np.array([1, 1, 1])
        self.Time_dist = np.array([1, 1, 1])
        self.MaxMag = 1000
        self.calib = np.array(0.0, dtype=np.float64)
        self.rho_ini = None  
        self.rho_fin = None  
        print("GlobalVars 初始化完成")
        # 预计算的X和Y分量算符（3量子比特，8x8）
        self.Hx = mykron(self.Ix, self.I, self.I) + \
                        mykron(self.I, self.Ix, self.I) + \
                        mykron(self.I, self.I, self.Ix)
                        
        self.Hy = mykron(self.Iy, self.I, self.I) + \
                        mykron(self.I, self.Iy, self.I) + \
                        mykron(self.I, self.I, self.Iy)
        self.B = None
        self.B_store =  {} 
        self.uwant = None
        self.pulsenumber = None
        self.user = 'Zidong Lin'
        self.calculatetype = 1
        self.File_Type = 2

# 张量积
def mykron(*args: np.ndarray) -> np.ndarray:
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

# 计算保真度函数
def calculate_fidelity(B, local_vars):
    # 计算脉冲序列的保真度
    H_int = local_vars.H_int
    Uideal = local_vars.Uideal
    M = local_vars.M    
    t = local_vars.t
    RF_deviation = local_vars.RF_deviation
    Spatial_dist = local_vars.Spatial_dist
    Time_dist = local_vars.Time_dist
    Hx = local_vars.Hx
    Hy = local_vars.Hy

    if B.ndim != 1:
        raise ValueError(f"B不是一维数组，维度为{B.ndim}")
    if B.size != 2 * M:
        raise ValueError(f"B的长度错误: {B.size}，应为{2 * M}")
    
    mean_fidelity_U = 0.0
    spatial_sum = np.sum(Spatial_dist)
    time_sum = np.sum(Time_dist)
    
    for h in range(len(Time_dist)):
        for l in range(len(RF_deviation)):
            U_current = np.eye(8, dtype=np.complex128)
            
            for j in range(M):
                H = H_int[h] + 2 * np.pi * RF_deviation[l] * (B[j] * Hx + B[j + M] * Hy)
                A = expm(-1j * t * H)
                U_current = A @ U_current
            
            fid = np.abs(np.trace(U_current @ np.conj(Uideal).T)) / 8
            mean_fidelity_U += fid * Spatial_dist[l] / spatial_sum * Time_dist[h] / time_sum
    
    return mean_fidelity_U

local_vars = GlobalVars()

# 脉冲优化线程函数
def optimize_pulse(params_queue, result_queue, fit_params, pulse_params, local_vars):
    try:
        # 提取参数
        w_F1, w_F2, w_F3 = fit_params['w_F1'], fit_params['w_F2'], fit_params['w_F3']
        J12, J13, J23 = fit_params['J12'], fit_params['J13'], fit_params['J23']
        pulsenumber, outputfile, M, t_pulse, gatetype, threshold, pulse_width, o1 = pulse_params[:8]
        calculatetype = pulse_params[9] 
        local_vars.calculatetype = calculatetype
        local_vars.pulsenumber = pulsenumber
        local_vars.M = M
        local_vars.t = t_pulse
        local_vars.calib = np.array(1 / (4 * pulse_width), dtype=np.float64)
        File_Type = local_vars.File_Type

        I = local_vars.I
        Ix, Iy, Iz = local_vars.Ix, local_vars.Iy, local_vars.Iz
        X, Y, Z = local_vars.X, local_vars.Y, local_vars.Z
        Hadamard = 1 / np.sqrt(2.0) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        sqrtX = 1 / np.sqrt(2.0) * np.array([[1, -1j], [-1j, 1]], dtype=np.complex128)
        sqrtY = 1 / np.sqrt(2.0) * np.array([[1, -1], [1, 1]], dtype=np.complex128)
        sqrtZ = 1 / np.sqrt(2.0) * np.array([[1-1j, 0], [0, 1+1j]], dtype=np.complex128)

        # 设置哈密顿量
        o1_drift = [-20, 0, 20]  # Hz

        # 构建H_int并验证维度
        local_vars.H_int = []
        for ii in range(len(o1_drift)):
            H = (2 * np.pi * (w_F1 + o1_drift[ii]) * mykron(Iz, I, I) +
                 2 * np.pi * (w_F2 + o1_drift[ii]) * mykron(I, Iz, I) +
                 2 * np.pi * (w_F3 + o1_drift[ii]) * mykron(I, I, Iz) +
                 2 * np.pi * J12 * (mykron(Ix, Ix, I) + 
                                  mykron(Iy, Iy, I) + 
                                  mykron(Iz, Iz, I)) +
                 2 * np.pi * J13 * (mykron(Ix, I, Ix) + 
                                  mykron(Iy, I, Iy) + 
                                  mykron(Iz, I, Iz)) +
                 2 * np.pi * J23 * (mykron(I, Ix, Ix) + 
                                  mykron(I, Iy, Iy) + 
                                  mykron(I, Iz, Iz)))
               
            local_vars.H_int.append(H)
        
        # 定义量子门
        def define_quantum_gates():
            SWAP = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.complex128)

            U_permute = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0]
            ], dtype=np.complex128)

            # 受控门
            U_CNOT12 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), X, I)
            
            U_CNOT21 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                        mykron(X, np.array([[0, 0], [0, 1]], dtype=np.complex128), I)
            
            U_CNOT13 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), I, X)
            
            U_CNOT31 = mykron(I, I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + \
                        mykron(X, I, np.array([[0, 0], [0, 1]], dtype=np.complex128))
            
            U_CNOT23 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                        mykron(I, np.array([[0, 0], [0, 1]], dtype=np.complex128), X)
            
            U_CNOT32 = mykron(I, I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + \
                        mykron(I, X, np.array([[0, 0], [0, 1]], dtype=np.complex128))

            # CCNO门
            U_CCNOT1 = (mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                        mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)) +
                        mykron(I, np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                        mykron(X, np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)))
            
            U_CCNOT2 = (mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                        mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, np.array([[0, 0], [0, 1]], dtype=np.complex128)) +
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), X, np.array([[0, 0], [0, 1]], dtype=np.complex128)))
            
            U_CCNOT3 = (mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128), I) +
                        mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128), I) +
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128), I) +
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128), X))

            # CZ门
            U_CZ12 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), Z, I)
            
            U_CZ21 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                    mykron(Z, np.array([[0, 0], [0, 1]], dtype=np.complex128), I)
            
            U_CZ90_12 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), sqrtZ, I)
            
            U_CZ13 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), I, Z)
            
            U_CZ31 = mykron(I, I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + \
                    mykron(Z, I, np.array([[0, 0], [0, 1]], dtype=np.complex128))
            
            U_CZ90_13 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                        mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), I, sqrtZ)
            
            U_CZ23 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                    mykron(I, np.array([[0, 0], [0, 1]], dtype=np.complex128), Z)
            
            U_CZ32 = mykron(I, I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + \
                    mykron(I, Z, np.array([[0, 0], [0, 1]], dtype=np.complex128))
            
            U_CZ90_23 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                        mykron(I, np.array([[0, 0], [0, 1]], dtype=np.complex128), sqrtZ)

            # CCZ门
            U_CCZ3 = (mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128), I) +
                    mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128), I) +
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128), I) +
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128), Z))
            
            U_CCZ2 = (mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                    mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, np.array([[0, 0], [0, 1]], dtype=np.complex128)) +
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), Z, np.array([[0, 0], [0, 1]], dtype=np.complex128)))
            
            U_CCZ1 = (mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                    mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)) +
                    mykron(I, np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[1, 0], [0, 0]], dtype=np.complex128)) +
                    mykron(Z, np.array([[0, 0], [0, 1]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)))

            # CY门
            U_CY12 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), -1j * Y, I)
            
            U_CY21 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                    mykron(-1j * Y, np.array([[0, 0], [0, 1]], dtype=np.complex128), I)
            
            U_CY13 = mykron(np.array([[1, 0], [0, 0]], dtype=np.complex128), I, I) + \
                    mykron(np.array([[0, 0], [0, 1]], dtype=np.complex128), I, -1j * Y)
            
            U_CY31 = mykron(I, I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + \
                    mykron(-1j * Y, I, np.array([[0, 0], [0, 1]], dtype=np.complex128))
            
            U_CY23 = mykron(I, np.array([[1, 0], [0, 0]], dtype=np.complex128), I) + \
                    mykron(I, np.array([[0, 0], [0, 1]], dtype=np.complex128), -1j * Y)
            
            U_CY32 = mykron(I, I, np.array([[1, 0], [0, 0]], dtype=np.complex128)) + \
                    mykron(I, -1j * Y, np.array([[0, 0], [0, 1]], dtype=np.complex128))

            hbar = (6.626e-34) / (2 * np.pi)
            KB = 1.3807e-23
            T_indoor = np.array(24 + 273.15, dtype=np.float64) 
            omega_F1 = o1 + w_F1
            omega_F2 = o1 + w_F2
            omega_F3 = o1 + w_F3
            sigma_F1 = hbar * omega_F1 / (KB * T_indoor)
            sigma_F2 = hbar * omega_F2 / (KB * T_indoor)
            sigma_F3 = hbar * omega_F3 / (KB * T_indoor)
            rho_eq_F1 = 0.5 * np.array([
                [1 + sigma_F1 / 2, 0],
                [0, 1 - sigma_F1 / 2]
            ], dtype=np.complex128)
            
            rho_eq_F2 = 0.5 * np.array([
                [1 + sigma_F2 / 2, 0],
                [0, 1 - sigma_F2 / 2]
            ], dtype=np.complex128)
            
            rho_eq_F3 = 0.5 * np.array([
                [1 + sigma_F3 / 2, 0],
                [0, 1 - sigma_F3 / 2]
            ], dtype=np.complex128)
            rho_eq = mykron(rho_eq_F1, rho_eq_F2, rho_eq_F3)
            
            # 有效密度矩阵
            diag_real = np.diag(rho_eq).real
            min_diag = np.min(diag_real)  
            I_3qubit = mykron(I, I, I)

            if min_diag < -1e-10:  # 仅当存在显著负对角元时（排除数值误差），才平移
                rho_shifted = rho_eq - min_diag * I_3qubit  # 减去最小对角元，确保对角元≥0
            else:  # 无负对角元（如均匀混合态），直接使用 rho_eq 作为 rho_shifted，无需平移
                rho_shifted = rho_eq.copy()

            tr_shifted = np.trace(rho_shifted).real  # 计算平移后矩阵的迹（取实部，避免复数误差）
            tr_shifted = max(tr_shifted, 1e-10)  # 避免除以0（数值稳定性：防止tr_shifted接近0）
            rho_eff = rho_shifted / tr_shifted
            local_vars.rho_ini = rho_eff
            beta_store = np.array([182.02, 179.04, 229.38, 193.46, 200.28, 105.75], dtype=np.float64) / 360 * 2 * np.pi
            # 预定义2x2矩阵
            mat_0001 = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1><1|
            mat_1000 = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0><0|
            term1 = beta_store[0] * mykron(Ix, mat_0001, mat_1000)
            term2 = beta_store[1] * mykron(mat_0001, Ix, mat_1000)
            term3 = beta_store[2] * mykron(mat_0001, mat_1000, Ix)
            term4 = beta_store[3] * mykron(mat_0001, Ix, mat_0001)
            term5 = beta_store[4] * mykron(Ix, mat_0001, mat_0001)
            term6 = beta_store[5] * mykron(mat_1000, Ix, mat_0001)
            U_PPS = expm(-1j * (term1 + term2 + term3 + term4 + term5 + term6))
            rho_fin = U_PPS @ local_vars.rho_ini @ U_PPS.conj().T
            local_vars.rho_fin = rho_fin 
            uwant = []
            # 1-7: Q1单量子比特门
            uwant.append(mykron(sqrtX, I, I))
            uwant.append(mykron(sqrtY, I, I))
            uwant.append(mykron(sqrtX.conj().T, I, I))
            uwant.append(mykron(sqrtY.conj().T, I, I))
            uwant.append(mykron(Hadamard, I, I))
            uwant.append(mykron(X, I, I))
            uwant.append(mykron(Y, I, I))
            
            # 8-14: Q2单量子比特门
            uwant.append(mykron(I, sqrtX, I))
            uwant.append(mykron(I, sqrtY, I))
            uwant.append(mykron(I, sqrtX.conj().T, I))
            uwant.append(mykron(I, sqrtY.conj().T, I))
            uwant.append(mykron(I, Hadamard, I))
            uwant.append(mykron(I, X, I))
            uwant.append(mykron(I, Y, I))
            
            # 15-21: Q3单量子比特门
            uwant.append(mykron(I, I, sqrtX))
            uwant.append(mykron(I, I, sqrtY))
            uwant.append(mykron(I, I, sqrtX.conj().T))
            uwant.append(mykron(I, I, sqrtY.conj().T))
            uwant.append(mykron(I, I, Hadamard))
            uwant.append(mykron(I, I, X))
            uwant.append(mykron(I, I, Y))

            # 22-27: CNOT门
            uwant.append(U_CNOT12)
            uwant.append(U_CNOT21)
            uwant.append(U_CNOT13)
            uwant.append(U_CNOT31)
            uwant.append(U_CNOT23)
            uwant.append(U_CNOT32)
            
            # 28-30: CZ门
            uwant.append(U_CZ12)
            uwant.append(U_CZ13)
            uwant.append(U_CZ23)
            
            # 31: CCZ门
            uwant.append(U_CCZ1)
            
            # 32-34: CCNOT门
            uwant.append(U_CCNOT1)
            uwant.append(U_CCNOT2)
            uwant.append(U_CCNOT3)
            
            # 35-36: PPS和置换门
            uwant.append(U_PPS)
            uwant.append(U_permute)
            
            # 37: 复合门
            uwant.append(U_CNOT23 @ U_CNOT21 @ mykron(I, Hadamard, I))
            
            # 38-40: SWAP门
            uwant.append(mykron(SWAP, I))
            uwant.append(np.array([
                [1,0,0,0,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,1,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,1]
            ], dtype=np.complex128))
            uwant.append(mykron(I, SWAP))
            
            # 41-43: 受控SWAP门
            uwant.append(mykron(np.array([[1,0],[0,0]], dtype=np.complex128), I, I) + mykron(np.array([[0,0],[0,1]], dtype=np.complex128), SWAP))
            uwant.append(np.array([
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,1]
            ], dtype=np.complex128))
            uwant.append(mykron(I, I, np.array([[1,0],[0,0]], dtype=np.complex128)) + mykron(SWAP, np.array([[0,0],[0,1]], dtype=np.complex128)))
            
            # 44-46: U_CZ90门
            uwant.append(U_CZ90_12)
            uwant.append(U_CZ90_13)
            uwant.append(U_CZ90_23)
            return uwant
        # 获取量子门
        local_vars.uwant = define_quantum_gates()

        # 基矢变换
        H_0 = (2 * np.pi * (w_F1 + o1) * mykron(Iz, I, I) +
            2 * np.pi * (w_F2 + o1) * mykron(I, Iz, I) +
            2 * np.pi * (w_F3 + o1) * mykron(I, I, Iz) +
            2 * np.pi * J12 * (mykron(Ix, Ix, I) + mykron(Iy, Iy, I) + mykron(Iz, Iz, I)) +
            2 * np.pi * J13 * (mykron(Ix, I, Ix) + mykron(Iy, I, Iy) + mykron(Iz, I, Iz)) +
            2 * np.pi * J23 * (mykron(I, Ix, Ix) + mykron(I, Iy, Iy) + mykron(I, Iz, Iz)))

        eig_vals, eig_vecs = nla.eig(H_0)
        eig_vals = np.real(eig_vals)
        diagnalelement = np.real(np.diag(H_0))
        diagonaltest = np.eye(8, dtype=H_0.dtype)
        for ii in range(8):
            diagonaltest[ii, ii] = diagnalelement[ii]
        eig_vals1, eig_vecs1 = nla.eig(diagonaltest)
        eig_vals1 = np.real(eig_vals1)
        eig_u = [eig_vecs[:, ii] for ii in range(8)]
        eig_u1 = [eig_vecs1[:, ii] for ii in range(8)]
        new_u = np.zeros_like(eig_vecs)
        new_u1 = np.zeros_like(eig_vecs1)
        for ii in range(8):
            for jj in range(8):
                pos_u = np.argmax(np.abs(eig_u[jj]))
                pos_u1 = np.argmax(np.abs(eig_u1[jj]))
                
                if pos_u == ii:
                    new_u[:, ii] = eig_u[jj]
                if pos_u1 == ii:
                    new_u1[:, ii] = eig_u1[jj]
        # 计算变换矩阵
        transform = new_u1 @ new_u.conj().T
        # 调整符号
        for ii in range(8):
            if np.real(transform[ii, ii]) < 0:
                transform[ii, :] = -transform[ii, :]
        # 计算理想酉矩阵
        target_gate = local_vars.uwant[pulsenumber-1]
        local_vars.Uideal = transform.conj().T @ target_gate @ transform
        if local_vars.calculatetype == 1:
            # 初始波形
            if gatetype == 1:
                randomnumber = max(1, M // 10)
                mean_fidelity_U = 0.0
                while mean_fidelity_U <= 0.2:
                    x = np.linspace(1, M, randomnumber)
                    xx = np.linspace(1, M, M)
                    Bx = (local_vars.MaxMag * np.random.rand(randomnumber) - local_vars.MaxMag / 2)
                    Bx[0] = Bx[-1] = 0.0
                    tck = splrep(x, Bx)
                    Bx_interp = splev(xx, tck)
                    By = (local_vars.MaxMag * np.random.rand(randomnumber) - local_vars.MaxMag / 2)
                    By[0] = By[-1] = 0.0
                    tck_by = splrep(x, By)
                    By_interp = splev(xx, tck_by)
                    B = np.concatenate([Bx_interp, By_interp])
                    mean_fidelity_U = calculate_fidelity(B, local_vars)
                print(f"随机初始脉冲生成成功，保真度：{mean_fidelity_U:.6f}")
                if local_vars.pulsenumber in [23, 28]:
                    B = local_vars.B_store.get(22, B)
                elif local_vars.pulsenumber in [25, 29, 31]:
                    B = local_vars.B_store.get(24, B)
                elif local_vars.pulsenumber in [27, 30]:
                    B = local_vars.B_store.get(26, B)
                elif local_vars.pulsenumber == 32:
                    B = local_vars.B_store.get(31, B)
                elif local_vars.pulsenumber in [33, 34, 35]:
                    B = local_vars.B_store.get(32, B)
                else:
                    B = B
                inputfidelity = mean_fidelity_U
            elif gatetype == 2:
                mean_fidelity_U = 0.0
                B_prev_np = local_vars.B_store.get(local_vars.pulsenumber - 1)
                if B_prev_np is None:
                    raise ValueError(f"前一个脉冲（编号{local_vars.pulsenumber - 1}）不存在，无法初始化当前脉冲")
                B = np.zeros(2 * M, dtype=np.float64)
                B_prev = B_prev_np
                amp = np.zeros(M, dtype=np.float64)
                phase = np.zeros(M, dtype=np.float64)
                for j in range(M):
                    complex_amp = complex(B_prev[j], B_prev[j + M]) 
                    amp[j] = np.abs(complex_amp) * 100 / np.abs(local_vars.calib)
                    phase_rad = np.angle(complex_amp)
                    phase_deg = phase_rad * 180 / np.pi
                    phase[j] = np.remainder(phase_deg, 360) + 90
                for j in range(M):
                    rad_phase = phase[j] / 360 * 2 * np.pi
                    B[j] = (amp[j] / 100) * np.abs(local_vars.calib) * np.cos(rad_phase)
                    B[j + M] = (amp[j] / 100) * np.abs(local_vars.calib) * np.sin(rad_phase)
                # 计算保真度
                mean_fidelity_U = calculate_fidelity(B, local_vars)
                inputfidelity = mean_fidelity_U
        elif local_vars.calculatetype == 2:
            if gatetype == 1:
                amp_np, phase_np, t_gatetype1 = dataout_SpinQ(
                    r'F:\OneClickGRAPE\{}\标准\{}'.format(local_vars.typename, local_vars.outputfile),
                    'load.spinq',
                    24,
                    local_vars.M
                )

                amp = amp_np  # 已为NumPy数组
                phase = phase_np  # 已为NumPy数组
                B = np.zeros(2 * M, dtype=np.float64)
                for j in range(local_vars.M):
                    rad_phase = phase[j] / 360 * 2 * np.pi
                    B[j] = (amp[j] / 100) * np.abs(local_vars.calib) * np.cos(rad_phase)
                    B[j + local_vars.M] = (amp[j] / 100) * np.abs(local_vars.calib) * np.sin(rad_phase)
                mean_fidelity_U = calculate_fidelity(B, local_vars)
                inputfidelity = mean_fidelity_U
                print(f"读取 .spinq 文件成功，脉冲保真度：{mean_fidelity_U:.6f}")
            elif gatetype == 2:
                prev_pulse_idx = local_vars.pulsenumber - 1
                B_prev_np = local_vars.B_store.get(prev_pulse_idx)
                
                if B_prev_np is None:
                    raise ValueError(
                        f"calculatetype==2 (gatetype==2) 错误：B_store 中无脉冲编号 {prev_pulse_idx} "
                        f"（当前脉冲编号：{local_vars.pulsenumber}），无法复用历史脉冲"
                    )
                B = np.zeros(2 * local_vars.M, dtype=np.float64)
                B_prev = B_prev_np
                amp = np.zeros(local_vars.M, dtype=np.float64)
                phase = np.zeros(local_vars.M, dtype=np.float64)
                for j in range(local_vars.M):
                    complex_amp = complex(B_prev[j], B_prev[j + local_vars.M])
                    amp[j] = np.abs(complex_amp) * 100 / np.abs(local_vars.calib)
                    phase_rad = np.angle(complex_amp)
                    phase_deg = phase_rad * 180 / np.pi
                    phase[j] = np.remainder(phase_deg, 360) + 90
                for j in range(local_vars.M):
                    rad_phase = phase[j] / 360 * 2 * np.pi
                    B[j] = (amp[j] / 100) * np.abs(local_vars.calib) * np.cos(rad_phase)
                    B[j + local_vars.M] = (amp[j] / 100) * np.abs(local_vars.calib) * np.sin(rad_phase)
                # 计算保真度
                mean_fidelity_U = calculate_fidelity(B, local_vars)
                inputfidelity = mean_fidelity_U
                print(f"复用历史脉冲（编号：{prev_pulse_idx}）成功，当前保真度：{mean_fidelity_U:.6f}")
        if mean_fidelity_U<threshold:
            randomnumber = max(1, local_vars.M // 10)
            fval = 0
            penalty = 0
            Sigma = local_vars.Sigma
            eq_constraint, bounds = get_fmincon_constraints(local_vars)
            iteration = 0
            max_iterations = 50
            target_reached = False
            while iteration < max_iterations and not target_reached:
                print(f"=== 开始第{iteration+1}/{max_iterations}次迭代 ===")
                fval_last=fval
                mean_fidelity_U_last=mean_fidelity_U
                result = minimize(
                    fun=lambda B: objective_wrapper(B, local_vars),
                    x0=np.real(B).astype(np.float64),  
                    jac=True,
                    method='trust-constr',
                    constraints=[eq_constraint],
                    bounds=bounds,
                    options={
                        'maxiter': 20,
                        'initial_tr_radius': 10.0,
                        'gtol': 1e-6,
                        'xtol': 1e-6,
                        'barrier_tol': 1e-6,
                        'verbose': 1
                    }
                )
                if hasattr(result, 'x') and len(result.x) == 2 * local_vars.M:
                    B_opt = result.x.astype(np.float64)
                    B = B_opt
                    print(f"已使用优化后的脉冲更新B，长度：{len(B)}")
                else:
                    print("result.x无效，未更新B")
                    iteration += 1  # 即使未更新，也计数（避免无限循环）
                    continue  # 跳过后续计算，直接进入下一次迭代

                # 计算优化后保真度
                actual_fidelity = calculate_fidelity(B, local_vars)
                print(f"优化后实际保真度：{actual_fidelity:.6f}")

                # 核心判断：若保真度达标，立即退出循环
                if actual_fidelity >= threshold:
                    print(f"保真度({actual_fidelity:.6f})已超过阈值({threshold})，提前退出循环")
                    target_reached = True  # 触发循环退出
                    break  # 立即退出while循环
                # 未达标则继续迭代
                iteration += 1
                print(f"保真度未达标，进入下一次迭代\n")

                if target_reached:
                    print(f"\n=== 循环正常结束：保真度达标 ===")
                else:
                    print(f"\n=== 循环结束：已达最大迭代次数（{max_iterations}），保真度未达标 ===")
                    print(f"最终实际保真度：{actual_fidelity:.6f}（阈值：{threshold}）")
                fval = result.fun
                real_part = B[:M]
                imag_part = B[M:]
                complex_amp = np.array(real_part, dtype=np.complex128) + 1j * np.array(imag_part, dtype=np.complex128)
                AMP = np.abs(complex_amp)
                PHASE = np.angle(complex_amp)
                penalty = 0.0
                mean_fidelity_U = (-fval / 1e7) + penalty
                if fval == fval_last and mean_fidelity_U > threshold:
                    break
                elif mean_fidelity_U == mean_fidelity_U_last and mean_fidelity_U < threshold:
                    mean_fidelity_U = 0
                    while mean_fidelity_U <= 0.2:
                        x = np.linspace(1, M, randomnumber)
                        xx = np.linspace(1, M, M)
                        Bx = (local_vars.MaxMag * np.random.rand(randomnumber) - local_vars.MaxMag / 2)
                        Bx[0] = Bx[-1] = 0.0
                        tck = splrep(x, Bx)
                        Bx_interp = splev(xx, tck)
                        By = (local_vars.MaxMag * np.random.rand(randomnumber) - local_vars.MaxMag / 2)
                        By[0] = By[-1] = 0.0
                        tck_by = splrep(x, By)
                        By_interp = splev(xx, tck_by)
                        B = np.concatenate([Bx_interp, By_interp])
                        mean_fidelity_U = calculate_fidelity(B, local_vars)
                        real_part = B[:M]
                        imag_part = B[M:]
                        complex_amp = np.array(real_part, dtype=np.complex128) + 1j * np.array(imag_part, dtype=np.complex128)
                        AMP = np.abs(complex_amp)
                        PHASE = np.angle(complex_amp)
                        iteration = 0
        num_H = len(local_vars.H_int)
        num_RF = len(local_vars.RF_deviation)
        U = [[np.eye(8, dtype=np.complex128) for _ in range(num_RF)] for __ in range(num_H)]
        for h in range(num_H):
            for l in range(num_RF):
                U_current = np.eye(8, dtype=np.complex128)
                for j in range(local_vars.M):
                    H = local_vars.H_int[h] + 2 * np.pi * local_vars.RF_deviation[l] * (
                        B[j] * local_vars.Hx + B[j + local_vars.M] * local_vars.Hy
                    )
                    A = expm(-1j * t_pulse * H)
                    U_current = A @ U_current  # 累积演化算符
                U[h][l] = U_current  # 保存实际演化算
        # 计算归一化因子
        normalization = calculate_state_fidelity(local_vars.rho_fin, local_vars.rho_fin)
        print(f"归一化因子: {normalization:.6f}")  # 理论上应为1.
        # 重构最终脉冲
        M = local_vars.M
        pulse = np.stack([B[:M], B[M:]], axis=1)
        # 计算振幅和相位（保留指定小数位）
        complex_amp = pulse[:, 0].astype(np.complex128) + 1j * pulse[:, 1].astype(np.complex128)
        amp_raw = np.abs(complex_amp)
        abs_calib = np.abs(local_vars.calib)
        amp_scaled = amp_raw * 100 / abs_calib
        amp = np.round(amp_scaled * 100) / 100  # 保留2位小数
        phase_rad = np.angle(complex_amp)
        phase_deg_raw = phase_rad * 180 / np.pi
        phase_deg_mod = np.remainder(phase_deg_raw, 360.0)
        phase = np.round(phase_deg_mod * 10000) / 10000  # 保留4位小数
        B_recon = np.zeros_like(B, dtype=np.float64)
        for j in range(M):
            phase_j_rad = phase[j] / 360.0 * 2 * np.pi
            B_recon[j] = (amp[j] / 100.0) * abs_calib * np.cos(phase_j_rad)
            B_recon[j + M] = (amp[j] / 100.0) * abs_calib * np.sin(phase_j_rad)
        rho0 = np.zeros((8, 8), dtype=np.complex128)
        rho0[0, 0] = 1.0
        rho1_grape = [[None for _ in range(num_RF)] for _ in range(num_H)]
        for h in range(num_H):
            for l in range(num_RF):
                U_h_l = U[h][l]
                rho1 = U_h_l @ rho0 @ U_h_l.conj().T
                rho1_grape[h][l] = rho1
        # 计算理想演化后状态
        rho1_ideal = local_vars.Uideal @ rho0 @ local_vars.Uideal.conj().T
        # 计算平均保真度
        mean_fidelity_state = 0.0
        mean_fidelity_U = 0.0  # 最终酉保真度（覆盖之前推导值）
        fidelity_state = np.zeros((num_H, num_RF))
        fidelity_U = np.zeros((num_H, num_RF))
        sum_spatial = np.sum(local_vars.Spatial_dist)
        sum_time = np.sum(local_vars.Time_dist)

        for h in range(num_H):
            for l in range(num_RF):
                # 状态保真度
                current_fid_state = calculate_state_fidelity(rho1_grape[h][l], rho1_ideal)
                # 酉保真度
                trace_val = np.trace(U[h][l] @ local_vars.Uideal.conj().T)
                current_fid_U = np.abs(trace_val) / 8.0  # 3量子比特，除以8
                # 加权系数
                weight = (local_vars.Spatial_dist[l] / sum_spatial) * (local_vars.Time_dist[h] / sum_time)
                
                # 累加平均保真度
                mean_fidelity_state += current_fid_state * weight
                mean_fidelity_U += current_fid_U * weight
        # 存储单个保真度
        for h in range(num_H):
            for l in range(num_RF):
                fidelity_state[h, l] = calculate_state_fidelity(rho1_grape[h][l], rho1_ideal)
                trace_val = np.trace(U[h][l] @ local_vars.Uideal.conj().T)
                fidelity_U[h][l] = np.abs(trace_val) / 8.0
        B_final_np = B_recon
        final_fidelity_U = mean_fidelity_U.item() if isinstance(mean_fidelity_U, np.ndarray) else mean_fidelity_U
        M = local_vars.M
        t_pulse = t_pulse  # 确保为NumPy标量或Python标量
        outputfile = outputfile
        local_vars = local_vars  

        if File_Type == 1:
            write_success = write_pulse_file(
                B_np=B_final_np,
                M=M,
                t_pulse=t_pulse,
                outputfile=outputfile,
                fidelity=final_fidelity_U,  # 传入最终酉保真度（与文件中的##GATE_FIDELITY对应）
                local_vars=local_vars
            )
        else:
            write_success = write_pulse_file_json(
                B_np=B_final_np,
                M=M,
                t_pulse=t_pulse,
                outputfile=outputfile,
                fidelity=final_fidelity_U,  # 传入最终酉保真度（与文件中的##GATE_FIDELITY对应）
                local_vars=local_vars
            )
        # 写入结果检查
        if write_success:
            print(f"优化完成！最终酉保真度: {final_fidelity_U:.6f}，文件已保存到 {outputfile}")
        else:
            print(f"优化完成，但文件写入失败")
        result_queue.put({
            'success': True,
            'pulsenumber': pulsenumber,
            'outputfile': outputfile,
            'fidelity_U': final_fidelity_U,
            'fidelity_state': mean_fidelity_state.item()
        })
            
    except Exception as e:
        result_queue.put({'success': False, 'error': str(e)})
        print(f"脉冲优化线程错误: {str(e)}")
        print(traceback.format_exc())

# 状态保真度计算
def calculate_state_fidelity(rho1: np.ndarray, rho2: np.ndarray) -> float:
    sqrt_rho1 = sqrtm(rho1)
    mid_mat = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_mid = sqrtm(mid_mat)
    fidelity = np.real(np.trace(sqrt_mid)) ** 2
    return min(float(fidelity), 1.0)

# 生成fmincon所需的约束
def get_fmincon_constraints(local_vars):
    M = local_vars.M
    num_vars = 2 * M
    Aeq = np.zeros((4, num_vars), dtype=np.float64) 
    Aeq[0, 0] = 1.0               
    Aeq[1, M-1] = 1.0             
    Aeq[2, M] = 1.0               
    Aeq[3, 2*M - 1] = 1.0         
    beq = np.zeros(4, dtype=np.float64)
    calib_abs = np.abs(local_vars.calib)  
    lb = -0.4 * calib_abs * np.ones(num_vars, dtype=np.float64)  # 下界
    ub = 0.4 * calib_abs * np.ones(num_vars, dtype=np.float64)   # 上界
    eq_constraint = LinearConstraint(Aeq, beq, beq)
    bounds = Bounds(lb, ub)
    return eq_constraint, bounds

def objective_wrapper(B, local_vars):
            """目标函数包装器：返回f和g，适配scipy minimize的jac=True参数"""
            f, g = Gradient_SpinQ_Unitary(B, local_vars)
            return f, g

def Gradient_SpinQ_Unitary(B, local_vars):
    M = local_vars.M    
    t = local_vars.t  
    H_int = local_vars.H_int  
    Uideal = local_vars.Uideal  
    RF_deviation = local_vars.RF_deviation  
    Spatial_dist = local_vars.Spatial_dist  
    Time_dist = local_vars.Time_dist  
    Hx = local_vars.Hx  
    Hy = local_vars.Hy  
    Sigma = local_vars.Sigma  
    f = 0.0
    g = np.zeros(2 * M, dtype=np.float64)  
    penalty = 0.0

    try:
        B_np = np.real(B).astype(np.float64)  # B为优化器输入的NumPy数组
        pulse = np.stack([B_np[:M], B_np[M:]], axis=1)
        complex_amp = pulse[:, 0].astype(np.complex128) + 1j * pulse[:, 1].astype(np.complex128)
        abs_calib = np.abs(local_vars.calib)
        amp_raw = np.abs(complex_amp) * 100 / abs_calib  # 对应MATLAB的 amp计算逻辑
        amp = np.round(amp_raw * 100) / 100 
        phase_rad = np.angle(complex_amp)
        phase_deg_raw = phase_rad * 180 / np.pi
        phase_deg_mod = np.remainder(phase_deg_raw, 360.0)  # 对应MATLAB mod
        phase = np.round(phase_deg_mod * 10000) / 10000 
        B_recon = np.zeros_like(B_np, dtype=np.float64)
        for j in range(M):
            phase_j_rad = phase[j] / 360.0 * 2 * np.pi  # 角度转弧度
            B_recon[j] = (amp[j] / 100.0) * abs_calib * np.cos(phase_j_rad)  # 重构Bx
            B_recon[j + M] = (amp[j] / 100.0) * abs_calib * np.sin(phase_j_rad)  # 重构By
        B_np = B_recon
        num_H = len(H_int)
        num_RF = len(RF_deviation)
        A = [[[np.zeros((8, 8), dtype=np.complex128) for _ in range(M)]
             for __ in range(num_RF)] for ___ in range(num_H)]
        U = [[np.eye(8, dtype=np.complex128) for _ in range(num_RF)]
             for __ in range(num_H)]
        for j in range(M):  # j: 0-based（对应MATLAB的1-based）
            Bx_j = B_np[j]        # 当前Bx值（实数）
            By_j = B_np[j + M]    # 当前By值（实数）
            for h in range(num_H):
                for l in range(num_RF):
                    H = H_int[h] + 2 * np.pi * RF_deviation[l] * (Bx_j * Hx + By_j * Hy)                   
                    A[h][l][j] = expm(-1j * t * H)  # -1j 是NumPy虚数表示
                    if j == 0:
                        U[h][l] = A[h][l][j].copy()  
                    else:
                        U[h][l] = A[h][l][j] @ U[h][l]  # NumPy矩阵乘法 @
        F_U = np.zeros((num_H, num_RF), dtype=np.float64)
        for h in range(num_H):
            for l in range(num_RF):
                trace_val = np.abs(np.trace(U[h][l] @ np.conj(Uideal).T))
                trace_val_clipped = np.clip(trace_val, 0.0, 8.0)
                fid_U = trace_val_clipped / 8.0
                F_U[h][l] = -1e7 * (fid_U - penalty)
        if M > 0 and num_H > 0 and num_RF > 0:
            G = np.zeros((num_H, num_RF, 2 * M), dtype=np.float64)
            for k in range(M):  # k: 脉冲段索引（对应MATLAB j=k+1）
                for h in range(num_H):
                    for l in range(num_RF):
                        if (k + 1) < M:
                            Pj = np.conj(A[h][l][k+1]).T  # 共轭转置
                            for jj in range(k+2, M):
                                Pj = Pj @ np.conj(A[h][l][jj]).T
                            Pj = Pj @ Uideal
                        elif (k + 1) == M:
                            Pj = np.conj(A[h][l][M-1]).T @ Uideal
                        else:
                            Pj = Uideal.copy()  # 复制避免修改原数组
                        if k == 0:
                            Xj = A[h][l][0].copy()
                        else:
                            Xj = A[h][l][0].copy()
                            for ii in range(1, k+1):
                                Xj = A[h][l][ii] @ Xj
                        term_x = np.conj(Pj).T @ (1j * t * 2 * np.pi * RF_deviation[l] * Hx) @ Xj
                        trace_term_x = np.trace(term_x)
                        trace_xy = np.trace(np.conj(Xj).T @ Pj)
                        G[h][l][k] = Sigma * 2 * np.real(trace_term_x * trace_xy) / 32
                        term_y = np.conj(Pj).T @ (1j * t * 2 * np.pi * RF_deviation[l] * Hy) @ Xj
                        trace_term_y = np.trace(term_y)
                        G[h][l][k + M] = Sigma * 2 * np.real(trace_term_y * trace_xy) / 32
            spatial_sum = np.sum(Spatial_dist)  
            time_sum = np.sum(Time_dist)
            spatial_sum = max(spatial_sum, 1e-10)
            time_sum = max(time_sum, 1e-10)

            for h in range(num_H):
                for l in range(num_RF):
                    weight = (Spatial_dist[l] / spatial_sum) * (Time_dist[h] / time_sum)
                    # 累积损失
                    f += F_U[h][l] * weight
                    # 累积梯度
                    for pp in range(2 * M):
                        g[pp] += G[h][l][pp] * weight
        f = np.real(f)
        if f > 0:
            f = -f  
        g = np.real(g)
        return f, g
    except Exception as e:
        print(f"Gradient_SpinQ_Unitary错误: {str(e)}")
        return np.real(f), np.real(g)

