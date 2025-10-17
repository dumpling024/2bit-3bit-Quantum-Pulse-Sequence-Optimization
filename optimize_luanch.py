from bit3.optimize import optimize_pulse, local_vars
import bit3.optimize as bit3_optimize
import json
import queue
import threading
import numpy as np
import time

"""主函数：设置参数并启动多线程处理"""
total_start_time = time.time()  # 总计时开始
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
pulse_result_queue = queue.Queue()
json_file = "fit_params.json"
with open(json_file, "r", encoding="utf-8") as f:
    fit_params = json.load(f)
M_Q1, t_Q1 = fit_params['M_Q1'], fit_params['t_Q1']  # 关键：t已为秒，无需转换
M_Q2, t_Q2 = fit_params['M_Q2'], fit_params['t_Q2']
M_Q3, t_Q3 = fit_params['M_Q3'], fit_params['t_Q3']
M_Q1Q2, t_Q1Q2 = fit_params['M_Q1Q2'], fit_params['t_Q1Q2']
M_Q1Q3, t_Q1Q3 = fit_params['M_Q1Q3'], fit_params['t_Q1Q3']
M_Q2Q3, t_Q2Q3 = fit_params['M_Q2Q3'], fit_params['t_Q2Q3']
threshold = fit_params['threshold']
typename = fit_params['typename']
File_Type = 1
local_vars.File_Type = File_Type
pulse_params_list = [
# 1-7: Q1单量子比特门（M=M_Q1, t=t_Q1）
(1, 'Q1_X90.spinq', M_Q1, t_Q1, 1,0.992, pulse_width, o1, Type, calculatetype),
(2, 'Q1_Y90.spinq', M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype),
(3, 'Q1_X90N.spinq', M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype),
(4, 'Q1_Y90N.spinq', M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype),
(5, 'Q1_H.spinq', M_Q1, t_Q1, 1, 0.992, pulse_width, o1, Type, calculatetype),
(6, 'Q1_X180.spinq', M_Q1, t_Q1, 1, 0.992, pulse_width, o1, Type, calculatetype),
(7, 'Q1_Y180.spinq', M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype),

# 8-14: Q2单量子比特门（M=M_Q2, t=t_Q2）
(8, 'Q2_X90.spinq', M_Q2, t_Q2, 1, 0.992, pulse_width, o1, Type, calculatetype),
(9, 'Q2_Y90.spinq', M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype),
(10, 'Q2_X90N.spinq', M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype),
(11, 'Q2_Y90N.spinq', M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype),
(12, 'Q2_H.spinq', M_Q2, t_Q2, 1, 0.992, pulse_width, o1, Type, calculatetype),
(13, 'Q2_X180.spinq', M_Q2, t_Q2, 1, 0.992, pulse_width, o1, Type, calculatetype),
(14, 'Q2_Y180.spinq', M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype),

# 15-21: Q3单量子比特门（M=M_Q3, t=t_Q3）
(15, 'Q3_X90.spinq', M_Q3, t_Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(16, 'Q3_Y90.spinq', M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype),
(17, 'Q3_X90N.spinq', M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype),
(18, 'Q3_Y90N.spinq', M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype),
(19, 'Q3_H.spinq', M_Q3, t_Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(20, 'Q3_X180.spinq', M_Q3, t_Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(21, 'Q3_Y180.spinq', M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype),

# 22-27: 两比特门（CNOT系列，M=M_Q1Q2/M_Q1Q3/M_Q2Q3, t对应值）
(22, 'CNOT12.spinq', M_Q1Q2, t_Q1Q2, 1, 0.992, pulse_width, o1, Type, calculatetype),
(23, 'CNOT21.spinq', M_Q1Q2, t_Q1Q2, 1, 0.992, pulse_width, o1, Type, calculatetype),
(24, 'CNOT13.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(25, 'CNOT31.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),  
(26, 'CNOT23.spinq', M_Q2Q3, t_Q2Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(27, 'CNOT32.spinq', M_Q2Q3, t_Q2Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),

# 28-30: CZ系列（M对应两比特门参数）
(28, 'CZ12.spinq', M_Q1Q2, t_Q1Q2, 1, 0.992, pulse_width, o1, Type, calculatetype),
(29, 'CZ13.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(30, 'CZ23.spinq', M_Q2Q3, t_Q2Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),  

# 31-34: 三比特门（CCZ/CCNOT系列）
(31, 'CCZ.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(32, 'CCNOT1.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(33, 'CCNOT2.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),
(34, 'CCNOT3.spinq', M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype),

# 35-36: PPS系列（阈值0.98）
(35, 'PPS_PART1.spinq', M_Q1Q3, t_Q1Q3, 1, 0.993, pulse_width, o1, Type, calculatetype),
(36, 'PPS_PART2.spinq', M_Q1Q3, t_Q1Q3, 1, 0.993, pulse_width, o1, Type, calculatetype),

# 37: GHZ门（阈值0.993）
(37, 'GHZ.spinq', M_Q2Q3, t_Q2Q3, 1, 0.993, pulse_width, o1, Type, calculatetype),

# 38-40: SWAP系列（阈值0.988）
(38, 'SWAP12.spinq', M_Q1Q2, t_Q1Q2, 1, 0.988, pulse_width, o1, Type, calculatetype),
(39, 'SWAP13.spinq', M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype),
(40, 'SWAP23.spinq', M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype),

# 41-43: CSWAP系列（阈值0.988）
(41, 'CSWAP1.spinq', M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype),
(42, 'CSWAP2.spinq', M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype),
(43, 'CSWAP3.spinq', M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype),
# 44-46: 3QSWAP系列（阈值0.988）
(44, 'CsqrtZ12.spinq', M_Q1Q2, t_Q1Q2, 1, 0.995, pulse_width, o1, Type, calculatetype),
(45, 'CsqrtZ13.spinq', M_Q1Q3, t_Q1Q3, 1, 0.995, pulse_width, o1, Type, calculatetype),
(46, 'CsqrtZ23.spinq', M_Q2Q3, t_Q2Q3, 1, 0.995, pulse_width, o1, Type, calculatetype)
]

# 启动脉冲优化线程
for idx, pulse_params in enumerate(pulse_params_list, 1):
    # 显示当前处理进度
    total_pulses = len(pulse_params_list)
    pulsenumber = pulse_params[0]
    outputfile = pulse_params[1]
    print(f"\n=== 正在处理第 {idx}/{total_pulses} 个脉冲 ===")
    print(f"脉冲编号: {pulsenumber}, 输出文件: {outputfile}")
    
    # 1. 创建单个线程（每次仅一个线程）
    # 注意：params_queue未实际使用，传入空队列不影响
    empty_queue = queue.Queue()
    thread = threading.Thread(
        target=optimize_pulse,
        args=(empty_queue, pulse_result_queue, fit_params, pulse_params, local_vars)
    )
    
    # 2. 启动线程并【等待线程完成】（关键：确保当前脉冲生成完再继续）
    thread.start()
    thread.join()  # 阻塞当前主线程，直到这个脉冲的优化线程结束
    
    # 3. 即时获取当前脉冲的结果（可选，便于实时查看结果）
    if not pulse_result_queue.empty():
        current_result = pulse_result_queue.get()
        if current_result['success']:
            print(f"第 {idx}/{total_pulses} 个脉冲生成成功！")
            print(f"  - 酉保真度: {current_result['fidelity_U']:.6f}")
            print(f"  - 状态保真度: {current_result['fidelity_state']:.6f}")
        else:
            print(f"第 {idx}/{total_pulses} 个脉冲生成失败: {current_result['error']}")

# --------------------------
# 所有脉冲处理完成后，汇总结果
# --------------------------
total_end_time = time.time()
total_time = total_end_time - total_start_time
print(f"\n所有脉冲优化完成，总耗时: {total_time:.2f}秒")
