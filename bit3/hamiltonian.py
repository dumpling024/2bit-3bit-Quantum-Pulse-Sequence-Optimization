import numpy as np
import scipy.linalg as la
from datetime import datetime
import traceback

def fit_hamiltonian_parameters(params_queue, fit_result_queue, P_exp1, o1, Type):
    """哈密顿量参数拟合线程"""
    try:
        # 初始化单量子比特算符
        I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
        Ix = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
        Iy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
        Iz = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2
        
        # 处理P_exp1中的零值
        epsilon = 1e-9
        P_exp1 = [p if p != 0 else epsilon for p in P_exp1]
        distance_store = 1e10
        P_exp1_np = np.array(P_exp1, dtype=np.float64)
        
        # 设置机型参数
        P_exp2 = {
            1: np.array([-1489, -1439, -1360, -1312, -68, 0, 59, 128, 954, 1004, 1023, 1072], dtype=np.float64),
            2: (np.array([-1293.8, -1242.26, -1163.67, -1111.786, -68.255, 0.81, 63.208, 131.207, 819.379, 870.99, 890.282, 941.643], dtype=np.float64) - 0.81),
            3: (np.array([-1048, -997, -919, -867, -84, -15, 45, 113, 621, 672, 689, 740], dtype=np.float64) + 15)
        }
        
        if Type == 1:
            P_exp2_selected = P_exp2[1]
            M_Q1, t_Q1 = 100, 30e-6
            M_Q2, t_Q2 = 100, 30e-6
            M_Q3, t_Q3 = 100, 30e-6
            M_Q1Q2, t_Q1Q2 = 300, 25e-6
            M_Q1Q3, t_Q1Q3 = 800, 25e-6
            M_Q2Q3, t_Q2Q3 = 600, 25e-6
            threshold = 0.995
            typename = 'Triangulum'
        
        elif Type == 2:
            P_exp2_selected = P_exp2[2]
            M_Q1, t_Q1 = 200, 25e-6
            M_Q2, t_Q2 = 200, 25e-6
            M_Q3, t_Q3 = 200, 25e-6
            M_Q1Q2, t_Q1Q2 = 400, 25e-6
            M_Q1Q3, t_Q1Q3 = 800, 25e-6
            M_Q2Q3, t_Q2Q3 = 600, 25e-6
            threshold = 0.995
            typename = 'Triangulumpro'
        
        elif Type == 3:
            P_exp2_selected = P_exp2[3]
            M_Q1, t_Q1 = 300, 35e-6
            M_Q2, t_Q2 = 300, 35e-6
            M_Q3, t_Q3 = 300, 35e-6
            M_Q1Q2, t_Q1Q2 = 300, 35e-6
            M_Q1Q3, t_Q1Q3 = 800, 35e-6
            M_Q2Q3, t_Q2Q3 = 600, 35e-6
            threshold = 0.994
            typename = 'Triangulummini'
        
        else:
            raise ValueError(f"无效机型Type: {Type}，必须为1、2或3")
        
        # 处理P_exp2中的零值
        P_exp2_selected = [p if p != 0 else epsilon for p in P_exp2_selected]

        # 计算初始中心值
        w1_center = ((P_exp1_np[0] + P_exp1_np[2])/2 + (P_exp1_np[1] + P_exp1_np[3])/2) / 2
        w2_center = ((P_exp1_np[4] + P_exp1_np[6])/2 + (P_exp1_np[5] + P_exp1_np[7])/2) / 2
        w3_center = ((P_exp1_np[8] + P_exp1_np[10])/2 + (P_exp1_np[9] + P_exp1_np[11])/2) / 2
        
        J12_center = -(abs(P_exp1_np[1] - P_exp1_np[3]) + abs(P_exp1_np[5] - P_exp1_np[7])) / 2
        J13_center = abs((P_exp1_np[0] + P_exp1_np[2])/2 - (P_exp1_np[1] + P_exp1_np[3])/2)
        J23_center = abs((P_exp1_np[4] + P_exp1_np[6])/2 - (P_exp1_np[5] + P_exp1_np[7])/2)
        
        # 计算multiple_center
        P_exp2_np = np.array(P_exp2_selected, dtype=np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = P_exp2_np / P_exp1_np
            valid_ratios = ratios[np.isfinite(ratios)]  # 筛选有效比例
            multiple_center = np.mean(valid_ratios) if len(valid_ratios) > 0 else 1.0  # 标量均值
        
        # 迭代优化
        range_val = 5
        w_F1_store, w_F2_store, w_F3_store = w1_center, w2_center, w3_center
        J_12_store, J_13_store, J_23_store = J12_center, J13_center, J23_center
        multiple_store = multiple_center
        
        for nn in range(5):
            # 创建参数网格
            w1 = np.linspace(w1_center - range_val, w1_center + range_val, 5)
            w2 = np.linspace(w2_center - range_val, w2_center + range_val, 5)
            w3 = np.linspace(w3_center - range_val, w3_center + range_val, 5)
            J12 = np.linspace(J12_center - range_val, J12_center + range_val, 5)
            J13 = np.linspace(J13_center - range_val, J13_center + range_val, 5)
            J23 = np.linspace(J23_center - range_val, J23_center + range_val, 5)
            multiple = np.linspace(multiple_center - 0.1e-3, multiple_center + 0.1e-3, 5)
            
            # 网格搜索
            for mm in range(len(multiple)):
                for aa in range(len(w1)):
                    for bb in range(len(w2)):
                        for cc in range(len(w3)):
                            for dd in range(len(J12)):
                                for ee in range(len(J13)):
                                    for ff in range(len(J23)):
                                        try:
                                            # 计算哈密顿量（3量子比特应为8x8）
                                            w_F1, w_F2, w_F3 = w1[aa], w2[bb], w3[cc]
                                            J_12, J_13, J_23 = J12[dd], J13[ee], J23[ff]
                                            multiple_B0 = multiple[mm]
                                            
                                            # 使用NumPy计算哈密顿量
                                            H_int1 = (2 * np.pi * (w_F1 + o1) * np.kron(np.kron(Iz, I), I) +
                                                     2 * np.pi * (w_F2 + o1) * np.kron(np.kron(I, Iz), I) +
                                                     2 * np.pi * (w_F3 + o1) * np.kron(np.kron(I, I), Iz) +
                                                     2 * np.pi * J_12 * (np.kron(np.kron(Ix, Ix), I) + np.kron(np.kron(Iy, Iy), I) + np.kron(np.kron(Iz, Iz), I)) +
                                                     2 * np.pi * J_13 * (np.kron(np.kron(Ix, I), Ix) + np.kron(np.kron(Iy, I), Iy) + np.kron(np.kron(Iz, I), Iz)) +
                                                     2 * np.pi * J_23 * (np.kron(np.kron(I, Ix), Ix) + np.kron(np.kron(I, Iy), Iy) + np.kron(np.kron(I, Iz), Iz)))
                                            
                                            if H_int1.shape != (8, 8):
                                                raise ValueError(f"H_int1形状错误: {H_int1.shape}，应为(8,8)")
                                            
                                            # 计算本征值
                                            eig_vals1, _ = la.eig(H_int1)
                                            eig_vals1 = np.real(eig_vals1)
                                            if len(eig_vals1) != 8:
                                                raise ValueError(f"H_int1本征值数量错误: {len(eig_vals1)}")
                                            
                                            # 计算P1
                                            P1 = np.zeros(12)
                                            P1[0] = eig_vals1[1] - eig_vals1[0]
                                            P1[1] = eig_vals1[7] - eig_vals1[6]
                                            P1[2] = eig_vals1[2] - eig_vals1[0]
                                            P1[3] = eig_vals1[4] - eig_vals1[2]
                                            P1[4] = eig_vals1[5] - eig_vals1[3]
                                            P1[5] = eig_vals1[7] - eig_vals1[5]
                                            P1[6] = eig_vals1[3] - eig_vals1[0]
                                            P1[7] = eig_vals1[4] - eig_vals1[1]
                                            P1[8] = eig_vals1[6] - eig_vals1[3]
                                            P1[9] = eig_vals1[7] - eig_vals1[4]
                                            P1[10] = eig_vals1[5] - eig_vals1[1]
                                            P1[11] = eig_vals1[6] - eig_vals1[2]
                                            
                                            # 构建H_int2
                                            H_int2 = (2 * np.pi * multiple_B0 * (w_F1 + o1) * np.kron(np.kron(Iz, I), I) +
                                                     2 * np.pi * multiple_B0 * (w_F2 + o1) * np.kron(np.kron(I, Iz), I) +
                                                     2 * np.pi * multiple_B0 * (w_F3 + o1) * np.kron(np.kron(I, I), Iz) +
                                                     2 * np.pi * J_12 * (np.kron(np.kron(Ix, Ix), I) + np.kron(np.kron(Iy, Iy), I) + np.kron(np.kron(Iz, Iz), I)) +
                                                     2 * np.pi * J_13 * (np.kron(np.kron(Ix, I), Ix) + np.kron(np.kron(Iy, I), Iy) + np.kron(np.kron(Iz, I), Iz)) +
                                                     2 * np.pi * J_23 * (np.kron(np.kron(I, Ix), Ix) + np.kron(np.kron(I, Iy), Iy) + np.kron(np.kron(I, Iz), Iz)))
                                            
                                            if H_int2.shape != (8, 8):
                                                raise ValueError(f"H_int2形状错误: {H_int2.shape}，应为(8,8)")
                                            
                                            # 计算本征值
                                            eig_vals2, _ = la.eig(H_int2)
                                            eig_vals2 = np.real(eig_vals2)
                                            if len(eig_vals2) != 8:
                                                raise ValueError(f"H_int2本征值数量错误: {len(eig_vals2)}")
                                            
                                            # 计算P2
                                            P2 = np.zeros(12)
                                            P2[0] = eig_vals2[1] - eig_vals2[0]
                                            P2[1] = eig_vals2[7] - eig_vals2[6]
                                            P2[2] = eig_vals2[2] - eig_vals2[0]
                                            P2[3] = eig_vals2[4] - eig_vals2[2]
                                            P2[4] = eig_vals2[5] - eig_vals2[3]
                                            P2[5] = eig_vals2[7] - eig_vals2[5]
                                            P2[6] = eig_vals2[3] - eig_vals2[0]
                                            P2[7] = eig_vals2[4] - eig_vals2[1]
                                            P2[8] = eig_vals2[6] - eig_vals2[3]
                                            P2[9] = eig_vals2[7] - eig_vals2[4]
                                            P2[10] = eig_vals2[5] - eig_vals2[1]
                                            P2[11] = eig_vals2[6] - eig_vals2[2]
                                            
                                            # 标准化并计算距离
                                            P1 = np.sort(P1) / (2 * np.pi)
                                            P1 = P1 - P1[5]
                                            P2 = np.sort(P2) / (2 * np.pi)
                                            P2 = P2 - P2[5]
                                            
                                            if np.any(np.isnan(P1)) or np.any(np.isinf(P1)):
                                                continue
                                            if np.any(np.isnan(P2)) or np.any(np.isinf(P2)):
                                                continue
                                            
                                            distance = 0
                                            for ii in range(len(P_exp1_np)):
                                                distance += abs(P_exp1_np[ii] - P1[ii]) + abs(P_exp2_np[ii] - P2[ii])
                                            
                                            # 更新最佳参数
                                            if distance < distance_store:
                                                distance_store = distance
                                                w_F1_store, w_F2_store, w_F3_store = w_F1, w_F2, w_F3
                                                J_12_store, J_13_store, J_23_store = J_12, J_13, J_23
                                                multiple_store = multiple_B0
                                                P_store1, P_store2 = P1.copy(), P2.copy()
                                        except Exception as e:
                                            print(f"参数组合计算错误: {str(e)}")
                                            continue
            
            # 更新中心值和范围
            w1_center, w2_center, w3_center = w_F1_store, w_F2_store, w_F3_store
            J12_center, J13_center, J23_center = J_12_store, J_13_store, J_23_store
            multiple_center = multiple_store
            range_val = 1  # 减小搜索范围
        
        # 返回拟合结果
        result = {
            '拟合时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'typename': typename,
            'threshold': threshold,
            'w_F1': w_F1_store,
            'w_F2': w_F2_store,
            'w_F3': w_F3_store,
            'J12': J_12_store,
            'J13': J_13_store,
            'J23': J_23_store,
            'multiple': multiple_store,
            'M_Q1': M_Q1, 't_Q1': t_Q1,
            'M_Q2': M_Q2, 't_Q2': t_Q2,
            'M_Q3': M_Q3, 't_Q3': t_Q3,
            'M_Q1Q2': M_Q1Q2, 't_Q1Q2': t_Q1Q2,
            'M_Q1Q3': M_Q1Q3, 't_Q1Q3': t_Q1Q3,
            'M_Q2Q3': M_Q2Q3, 't_Q2Q3': t_Q2Q3,
            '最小误差': distance_store
        }
        fit_result_queue.put(result)  # 结果放入队列
        
    except Exception as e:
        fit_result_queue.put(None)
        print(f"参数拟合线程错误: {str(e)}")
        print(traceback.format_exc())
