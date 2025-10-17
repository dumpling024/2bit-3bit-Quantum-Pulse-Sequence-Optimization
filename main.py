import sys
import os
import json
import time
import threading
import queue
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QListWidget,
    QListWidgetItem, QMessageBox, QGroupBox, QGridLayout, QDoubleSpinBox,
    QRadioButton, QButtonGroup, QTextEdit, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont
import io

# 导入后端模块
from bit3.hamiltonian import *
import bit3.optimize as bit3_optimize
import bit2.OneClickSMP as bit2_optimize
from bit3.write import *
import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TORCHDYNAMO_DISABLE"] = "1"
thread_lock = threading.Lock()

# 3bit后端任务线程
class Bit3BackendThread(QThread):
    finish_signal = pyqtSignal(bool, str)  # (是否成功, 结果信息)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def run(self):
        try:
            total_start_time = time.time() 
            # 提取基础参数
            pulse_width = float(self.params["pulse_width"])
            P_exp1_str = self.params["P_exp1"]
            o1 = float(self.params["o1"])
            Type = int(self.params["Type"])
            calculatetype = int(self.params["calculatetype"])
            fit_file_path = self.params["fit_file_path"]
            File_Type = int(self.params["File_Type"])

            output_dir = self.params["output_dir"]
            selected_pulses = self.params["selected_pulses"]

            # 处理P_exp1数组
            try:
                P_exp1 = np.array([float(x.strip()) for x in P_exp1_str.split(",")])
                try:
                    adjust_value = float(self.params["P_exp1_adjust"])
                except:
                    adjust_value = -8
                P_exp1 = P_exp1 + adjust_value
                if len(P_exp1) < 12:
                    default_P_exp1 = np.array([-1280, -1225, -1149, -1094, -60, 8, 74, 140, 822, 873, 893, 946]) + (-8)
                    P_exp1 = np.pad(P_exp1, (0, 12 - len(P_exp1)), mode="constant", constant_values=default_P_exp1[len(P_exp1):])
                elif len(P_exp1) > 12:
                    P_exp1 = P_exp1[:12]
            except Exception as e:
                P_exp1 = np.array([-1280, -1225, -1149, -1094, -60, 8, 74, 140, 822, 873, 893, 946]) + (-8)

            if not fit_file_path:
                params_queue = queue.Queue()
                fit_result_queue = queue.Queue()
                fit_thread = threading.Thread(
                    target=fit_hamiltonian_parameters,
                    args=(params_queue, fit_result_queue, P_exp1, o1, Type)
                )
                fit_thread.start()
                fit_thread.join()
                fit_params = fit_result_queue.get()
                if fit_params is not None:
                    fit_file_path = os.path.join(output_dir, "fit_params.json")
                    save_fit_result_to_file(fit_params, filename=fit_file_path)
                else:
                    raise Exception("拟合线程返回空结果")
            else:
                with open(fit_file_path, "r", encoding="utf-8") as f:
                    fit_params = json.load(f)

            # 解析拟合参数
            M_Q1, t_Q1 = fit_params['M_Q1'], fit_params['t_Q1']
            M_Q2, t_Q2 = fit_params['M_Q2'], fit_params['t_Q2']
            M_Q3, t_Q3 = fit_params['M_Q3'], fit_params['t_Q3']
            M_Q1Q2, t_Q1Q2 = fit_params['M_Q1Q2'], fit_params['t_Q1Q2']
            M_Q1Q3, t_Q1Q3 = fit_params['M_Q1Q3'], fit_params['t_Q1Q3']
            M_Q2Q3, t_Q2Q3 = fit_params['M_Q2Q3'], fit_params['t_Q2Q3']
            threshold = fit_params['threshold']
            typename = fit_params['typename']
            
            # 构建3比特脉冲列表
            pulse_params_list = []
            pulse_id = 1
            if "Q1_X90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_X90.spinq"), M_Q1, t_Q1, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q1_Y90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_Y90.spinq"), M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q1_X90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_X90N.spinq"), M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q1_Y90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_Y90N.spinq"), M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q1_H" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_H.spinq"), M_Q1, t_Q1, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q1_X180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_X180.spinq"), M_Q1, t_Q1, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q1_Y180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q1_Y180.spinq"), M_Q1, t_Q1, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 2. Q2单量子比特门（8-14号） --------------------------
            if "Q2_X90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_X90.spinq"), M_Q2, t_Q2, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q2_Y90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_Y90.spinq"), M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q2_X90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_X90N.spinq"), M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q2_Y90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_Y90N.spinq"), M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q2_H" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_H.spinq"), M_Q2, t_Q2, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q2_X180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_X180.spinq"), M_Q2, t_Q2, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q2_Y180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q2_Y180.spinq"), M_Q2, t_Q2, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 3. Q3单量子比特门（15-21号） --------------------------
            if "Q3_X90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_X90.spinq"), M_Q3, t_Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q3_Y90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_Y90.spinq"), M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q3_X90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_X90N.spinq"), M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q3_Y90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_Y90N.spinq"), M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q3_H" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_H.spinq"), M_Q3, t_Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q3_X180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_X180.spinq"), M_Q3, t_Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "Q3_Y180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Q3_Y180.spinq"), M_Q3, t_Q3, 2, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 4. 两比特门-CNOT系列（22-27号） --------------------------
            if "CNOT12" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CNOT12.spinq"), M_Q1Q2, t_Q1Q2, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CNOT21" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CNOT21.spinq"), M_Q1Q2, t_Q1Q2, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CNOT13" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CNOT13.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CNOT31" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CNOT31.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CNOT23" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CNOT23.spinq"), M_Q2Q3, t_Q2Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CNOT32" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CNOT32.spinq"), M_Q2Q3, t_Q2Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 5. 两比特门-CZ系列（28-30号） --------------------------
            if "CZ12" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CZ12.spinq"), M_Q1Q2, t_Q1Q2, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CZ13" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CZ13.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CZ23" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CZ23.spinq"), M_Q2Q3, t_Q2Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 6. 三比特门-CCZ/CCNOT系列（31-34号） --------------------------
            if "CCZ" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CCZ.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CCNOT1" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CCNOT1.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CCNOT2" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CCNOT2.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CCNOT3" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CCNOT3.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.992, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 7. PPS系列（35-36号，阈值0.993） --------------------------
            if "PPS_PART1" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "PPS_PART1.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.993, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "PPS_PART2" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "PPS_PART2.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.993, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 8. 三比特门-GHZ（37号，阈值0.993） --------------------------
            if "GHZ" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "GHZ.spinq"), M_Q2Q3, t_Q2Q3, 1, 0.993, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 9. 两比特门-SWAP系列（38-40号，阈值0.988） --------------------------
            if "SWAP12" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "SWAP12.spinq"), M_Q1Q2, t_Q1Q2, 1, 0.988, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "SWAP13" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "SWAP13.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "SWAP23" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "SWAP23.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 10. 三比特门-CSWAP系列（41-43号，阈值0.988） --------------------------
            if "CSWAP1" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CSWAP1.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CSWAP2" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CSWAP2.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CSWAP3" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CSWAP3.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.988, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1

            # -------------------------- 11. 两比特门-CsqrtZ系列（44-46号，阈值0.995） --------------------------
            if "CsqrtZ12" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CsqrtZ12.spinq"), M_Q1Q2, t_Q1Q2, 1, 0.995, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CsqrtZ13" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CsqrtZ13.spinq"), M_Q1Q3, t_Q1Q3, 1, 0.995, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1
            if "CsqrtZ23" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "CsqrtZ23.spinq"), M_Q2Q3, t_Q2Q3, 1, 0.995, pulse_width, o1, Type, calculatetype)
                )
                pulse_id += 1            
            if not pulse_params_list:
                self.finish_signal.emit(False, "未选择脉冲类型")
                return
            bit3_optimize.local_vars.File_Type = File_Type
            pulse_result_queue = queue.Queue()
            for idx, pulse_params in enumerate(pulse_params_list, 1):
                total_pulses = len(pulse_params_list)
                pulsenumber = pulse_params[0]
                outputfile = pulse_params[1]
                print(f"\n=== 3比特：正在处理第 {idx}/{total_pulses} 个脉冲 ===")
                print(f"脉冲编号: {pulsenumber}, 输出文件: {outputfile}")
                empty_queue = queue.Queue()
                thread = threading.Thread(
                    target=bit3_optimize.optimize_pulse,
                    args=(empty_queue, pulse_result_queue, fit_params, pulse_params, bit3_optimize.local_vars)
                )
                thread.start()
                thread.join()
                if not pulse_result_queue.empty():
                        current_result = pulse_result_queue.get()
                        if current_result['success']:
                            print(f"第 {idx}/{total_pulses} 个脉冲生成成功！")
                            print(f"  - 酉保真度: {current_result['fidelity_U']:.6f}")
                            print(f"  - 状态保真度: {current_result['fidelity_state']:.6f}")
                        else:
                            print(f"第 {idx}/{total_pulses} 个脉冲生成失败: {current_result['error']}")

            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            print(f"\n3比特：所有脉冲优化完成，总耗时: {total_time:.2f}秒")
            self.finish_signal.emit(True, f"共处理 {len(pulse_params_list)} 个脉冲，总耗时 {total_time:.2f}秒")
        except Exception as e:
            error_msg = f"3比特任务异常：{str(e)}\n{traceback.format_exc()}"
            self.finish_signal.emit(False, error_msg)

    def stop(self):
        self.is_running = False

# 2bit后端任务线程
class Bit2BackendThread(QThread):
    finish_signal = pyqtSignal(bool, str)  # (是否成功, 结果信息)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.is_running = True

    def run(self):
        try:
            total_start_time = time.time()
            # 提取基础参数
            P1_methyl = float(self.params["P1_methyl"])
            P2_methyl = float(self.params["P2_methyl"])
            w_H = float(self.params["w_H"]) * 10**6  
            w_P = float(self.params["w_P"]) * 10**6  
            PW_H = float(self.params["PW_H"]) * 10**-6  
            PW_P = float(self.params["PW_P"]) * 10**-6  
            T_indoor = float(self.params["T_indoor"]) + 273.15  

            w_1 = float(self.params["w_1"])
            w_2 = float(self.params["w_2"])
            J12 = float(self.params["J12"])
            w_3 = (P1_methyl + P2_methyl) / 2
            w_4 = float(self.params["w_4"])
            J23 = abs(P1_methyl - P2_methyl)
            o1 = list(map(float, self.params["o1"].split(",")))
            o2 = [x * w_P / w_H for x in o1]
            pulse_width_H = np.array([0.95, 1, 1.05]) * PW_H
            pulse_width_P = np.array([0.95, 1, 1.05]) * PW_P
            File_Type = int(self.params["File_Type"])

            output_dir = self.params["output_dir"]
            selected_pulses = self.params["selected_pulses"]

            fit_params = (
                PW_H, PW_P, T_indoor, w_H, w_P, J12, J23,
                w_1, w_2, w_3, w_4, o1, o2, pulse_width_H, pulse_width_P, File_Type
            )

            pulse_params_list = []
            pulse_id = 1
            if "Gemini_Q1_X90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_X90.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q1_Y90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_Y90.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q1_X90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_X90N.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q1_Y90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_Y90N.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_X90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_X90.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_Y90" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_Y90.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_X90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_X90N.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_Y90N" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_Y90N.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q1_X180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_X180.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q1_Y180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_Y180.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_X180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_X180.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_Y180" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_Y180.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q1_H" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q1_H.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_Q2_H" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_Q2_H.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_CNOT12" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_CNOT12.spinq"), 0.9996)
                )
                pulse_id += 1
            if "Gemini_CNOT21" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_CNOT21.spinq"), 0.999)
                )
                pulse_id += 1
            if "Gemini_CZ" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_CZ.spinq"), 0.999)
                )
                pulse_id += 1
            if "Gemini_SWAP" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "Gemini_SWAP.spinq"), 0.999)
                )
                pulse_id += 1
            if "PPS_PART1" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "PPS_PART1.spinq"), 0.998)
                )
                pulse_id += 1
            if "PPS_PART2" in selected_pulses:
                pulse_params_list.append(
                    (pulse_id, os.path.join(output_dir, "PPS_PART2.spinq"), 0.998)
                )
                pulse_id += 1

            if not pulse_params_list:
                self.finish_signal.emit(False, "未选择脉冲类型")
                return

            pulse_result_queue = queue.Queue()
            for idx, pulse_params in enumerate(pulse_params_list, 1):
                total_pulses = len(pulse_params_list)
                pulsenumber = pulse_params[0]
                outputfile = pulse_params[1]
                bit2_optimize.local_vars.pulsenumber = pulsenumber
                print(f"\n=== 2比特：正在处理第 {idx}/{total_pulses} 个脉冲 ===")
                print(f"脉冲编号: {pulsenumber}, 输出文件: {outputfile}")

                empty_queue = queue.Queue()
                thread = threading.Thread(
                    target=bit2_optimize.optimize_pulse,
                    args=(empty_queue, pulse_result_queue, fit_params, pulse_params, bit2_optimize.local_vars)
                )
                thread.start()
                thread.join()
            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            print(f"\n2比特：所有脉冲优化完成，总耗时: {total_time:.2f}秒")
            self.finish_signal.emit(True, f"共处理 {len(pulse_params_list)} 个脉冲，总耗时 {total_time:.2f}秒")

        except Exception as e:
            error_msg = f"2比特任务异常：{str(e)}\n{traceback.format_exc()}"
            self.finish_signal.emit(False, error_msg)

    def stop(self):
        self.is_running = False

# 3.1 3比特界面类
class Bit3PulseUI(QWidget):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.backend_thread = None
        self.original_stdout = sys.stdout
        self.log_stream = self.LogStream(self.log_signal)
        self.init_ui()

    class LogStream(io.TextIOBase):
        def __init__(self, log_signal):
            super().__init__()
            self.log_signal = log_signal

        def write(self, text):
            if text.strip():  
                self.log_signal.emit(text)

        def flush(self):
            pass  # 避免输出缓存问题

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 1. 标题
        title_label = QLabel("3比特量子计算机脉冲序列生成器")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 2. 参数输入区
        param_group = QGroupBox("3比特基础参数配置")
        param_group.setFont(QFont("微软雅黑", 11))
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(12)
        param_layout.setContentsMargins(15, 15, 15, 15)

        # 2.1 脉宽输入
        param_layout.addWidget(QLabel("90°脉宽 (s)："), 0, 0)
        self.pulse_width_edit = QLineEdit("30e-6")
        self.pulse_width_edit.setPlaceholderText("例如：30e-6（默认30微秒）")
        param_layout.addWidget(self.pulse_width_edit, 0, 1)

        # 2.2 P_exp1输入
        param_layout.addWidget(QLabel("P_exp1（12个值）："), 1, 0)
        self.P_exp1_edit = QLineEdit("-1280, -1225, -1149, -1094, -60, 8, 74, 140, 822, 873, 893, 946")
        self.P_exp1_edit.setPlaceholderText("例如：-1280,-1225,...（三角座mini默认值）")
        param_layout.addWidget(self.P_exp1_edit, 1, 1)
        param_layout.addWidget(QLabel("统一调整值："), 1, 2)
        self.P_exp1_adjust = QLineEdit("-8")
        self.P_exp1_adjust.setPlaceholderText("如-8、+10")
        self.P_exp1_adjust.setMaximumWidth(100)
        param_layout.addWidget(self.P_exp1_adjust, 1, 3)
        self.P_exp1_import_btn = QPushButton("导入TXT")
        self.P_exp1_import_btn.clicked.connect(self.import_P_exp1)
        param_layout.addWidget(self.P_exp1_import_btn, 1, 4)

        # 2.3 共振频率o1
        param_layout.addWidget(QLabel("共振频率o1 (Hz)："), 2, 0)
        self.o1_edit = QLineEdit("33832900.0")
        self.o1_edit.setPlaceholderText("例如：33832900.0（三角座mini默认）")
        param_layout.addWidget(self.o1_edit, 2, 1)

        # 2.4 机型选择
        param_layout.addWidget(QLabel("机型选择："), 3, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["1-三角座", "2-三角座Pro", "3-三角座Mini"])
        self.type_combo.setCurrentIndex(2)
        param_layout.addWidget(self.type_combo, 3, 1)

        # 2.5 优化类型选择
        param_layout.addWidget(QLabel("优化类型："), 4, 0)
        self.calc_type_combo = QComboBox()
        self.calc_type_combo.addItems(["1-新建脉冲文件", "2-基于旧文件优化"])
        self.calc_type_combo.setCurrentIndex(0)
        param_layout.addWidget(self.calc_type_combo, 4, 1)

        # 2.6 拟合文件选择
        param_layout.addWidget(QLabel("拟合结果文件："), 5, 0)
        self.fit_file_edit = QLineEdit()
        self.fit_file_edit.setPlaceholderText("可选：已有fit_params.json路径")
        param_layout.addWidget(self.fit_file_edit, 5, 1)
        self.fit_file_btn = QPushButton("选择文件")
        self.fit_file_btn.clicked.connect(self.select_fit_file)
        param_layout.addWidget(self.fit_file_btn, 5, 2)

        # 2.7 文件类型选择
        param_layout.addWidget(QLabel("输出文件类型："), 6, 0)
        self.file_type_group = QButtonGroup(self)
        self.spinq_radio = QRadioButton("spinq文件")
        self.spinq_radio.setChecked(True)
        self.json_radio = QRadioButton("json文件")
        self.file_type_group.addButton(self.spinq_radio, 1)
        self.file_type_group.addButton(self.json_radio, 0)
        file_type_layout = QHBoxLayout()
        file_type_layout.addWidget(self.spinq_radio)
        file_type_layout.addWidget(self.json_radio)
        param_layout.addLayout(file_type_layout, 6, 1)

        # 2.8 输出目录选择
        param_layout.addWidget(QLabel("脉冲输出目录："), 7, 0)
        self.output_dir_edit = QLineEdit(os.getcwd())
        self.output_dir_edit.setPlaceholderText("脉冲文件保存路径")
        param_layout.addWidget(self.output_dir_edit, 7, 1)
        self.output_dir_btn = QPushButton("选择目录")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        param_layout.addWidget(self.output_dir_btn, 7, 2)

        main_layout.addWidget(param_group)

        # 3. 脉冲类型选择区
        pulse_group = QGroupBox("3比特脉冲类型选择（勾选需要生成的脉冲）")
        pulse_group.setFont(QFont("微软雅黑", 11))
        pulse_layout = QHBoxLayout(pulse_group)
        self.pulse_list = QListWidget()
        pulse_types = [
            "Q1_X90", "Q1_Y90", "Q1_X90N", "Q1_Y90N", "Q1_H", "Q1_X180", "Q1_Y180", 
            "Q2_X90", "Q2_Y90", "Q2_X90N", "Q2_Y90N", "Q2_H", "Q2_X180", "Q2_Y180",
            "Q3_X90", "Q3_Y90", "Q3_X90N", "Q3_Y90N", "Q3_H", "Q3_X180", "Q3_Y180",
            "CNOT12", "CNOT21", "CNOT13", "CNOT31", "CNOT23", "CNOT32",
            "CZ12", "CZ13", "CZ23", "CCZ", "CCNOT1", "CCNOT2", "CCNOT3", 
            "PPS_PART1", "PPS_PART2", "GHZ", "SWAP12", "SWAP13", "SWAP23",
            "CSWAP1", "CSWAP2", "CSWAP3", "CsqrtZ12", "CsqrtZ13", "CsqrtZ23"
        ]
        for ptype in pulse_types:
            item = QListWidgetItem(ptype)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.pulse_list.addItem(item)
        pulse_layout.addWidget(self.pulse_list)
        btn_layout = QVBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_pulses)
        self.deselect_all_btn = QPushButton("取消全选")
        self.deselect_all_btn.clicked.connect(self.deselect_all_pulses)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        btn_layout.addStretch()
        pulse_layout.addLayout(btn_layout)
        main_layout.addWidget(pulse_group)

        log_group = QGroupBox("3比特运行日志（终端print信息）")
        log_group.setFont(QFont("微软雅黑", 11))
        log_layout = QVBoxLayout(log_group)

        # 日志显示控件
        self.log_textedit = QTextEdit()
        self.log_textedit.setReadOnly(True)
        self.log_textedit.setLineWrapMode(QTextEdit.NoWrap)  # 日志不换行，格式清晰
        self.log_textedit.setFont(QFont("Consolas", 10))  # 等宽字体，对齐输出
        self.log_textedit.setPlaceholderText("任务启动后，终端print信息将显示在这里...")

        # 清空日志按钮
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)

        # 日志布局：按钮在上，日志框在下（日志框占满剩余空间）
        log_btn_layout = QHBoxLayout()
        log_btn_layout.addWidget(self.clear_log_btn)
        log_btn_layout.addStretch()  # 按钮靠左
        log_layout.addLayout(log_btn_layout)
        log_layout.addWidget(self.log_textedit)

        main_layout.addWidget(log_group)

        # 4. 控制按钮区
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        self.start_btn = QPushButton("开始生成")
        self.start_btn.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_generation)
        self.stop_btn = QPushButton("停止任务")
        self.stop_btn.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        self.log_signal.connect(self.update_log)

    @pyqtSlot(str)
    def update_log(self, text):
        """槽函数：更新日志到QTextEdit（主线程执行，避免UI冲突）"""
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_textedit.append(timestamp + text)
        # 自动滚动到底部，查看最新日志
        self.log_textedit.moveCursor(self.log_textedit.textCursor().End)
        # 限制日志最大行数（避免内存占用过大，设为1000行）
        if self.log_textedit.document().blockCount() > 1000:
            self.log_textedit.clear()
            self.log_textedit.append("[日志已清空] 超过1000行，自动清理历史日志")

    def clear_log(self):
        """手动清空日志"""
        self.log_textedit.clear()
        self.log_textedit.append("[日志已清空] 手动清理完成")

    # ------------------------------ 界面功能函数（完全保留原逻辑） ------------------------------
    def import_P_exp1(self):
        """导入P_exp1的TXT文件（每行一个数值，共12行），自动应用统一调整值"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择P_exp1文件", "", "TXT文件 (*.txt)")
        if file_path:
            try:
                with open(file_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if len(lines) >= 1:
                        p_exp1_list = [float(x) for x in lines[:12]]
                        try:
                            adjust_value = float(self.P_exp1_adjust.text().strip())
                        except (ValueError, AttributeError):
                            adjust_value = -8
                        p_exp1_list = [x + adjust_value for x in p_exp1_list]
                        P_exp1_str = ",".join([str(round(x, 2)) for x in p_exp1_list])
                        self.P_exp1_edit.setText(P_exp1_str)
                    else:
                        QMessageBox.warning(self, "警告", "文件中无有效数值！")
            except Exception as e:
                QMessageBox.error(self, "导入失败", f"错误原因：{str(e)}")

    def select_fit_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择拟合结果文件", "", "JSON文件 (*.json)")
        if file_path:
            self.fit_file_edit.setText(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", os.getcwd())
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def select_all_pulses(self):
        for i in range(self.pulse_list.count()):
            item = self.pulse_list.item(i)
            item.setCheckState(Qt.Checked)

    def deselect_all_pulses(self):
        for i in range(self.pulse_list.count()):
            item = self.pulse_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def get_selected_pulses(self):
        """获取用户勾选的脉冲类型列表"""
        selected = []
        for i in range(self.pulse_list.count()):
            item = self.pulse_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected

    def start_generation(self):
        output_dir = self.output_dir_edit.text().strip()
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "目录不存在", "输出目录不存在，将自动创建！")
            os.makedirs(output_dir, exist_ok=True)

        selected_pulses = self.get_selected_pulses()
        if not selected_pulses:
            QMessageBox.warning(self, "未选择脉冲", "请至少勾选一种需要生成的脉冲类型！")
            return

        params = {
            "pulse_width": self.pulse_width_edit.text().strip(),
            "P_exp1": self.P_exp1_edit.text().strip(),
            "P_exp1_adjust": self.P_exp1_adjust.text().strip(),
            "o1": self.o1_edit.text().strip(),
            "Type": int(self.type_combo.currentText().split("-")[0]),
            "calculatetype": int(self.calc_type_combo.currentText().split("-")[0]),
            "fit_file_path": self.fit_file_edit.text().strip(),
            "output_dir": output_dir,
            "selected_pulses": selected_pulses,
            "start_time": time.time(),
            "File_Type": self.file_type_group.checkedId()
        }

        sys.stdout = self.log_stream
        print(f"3比特任务启动，选中脉冲数量：{len(selected_pulses)}")

        self.backend_thread = Bit3BackendThread(params)
        self.backend_thread.finish_signal.connect(self.on_task_finish)
        self.backend_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_generation(self):
        if self.backend_thread and self.backend_thread.isRunning():
            self.backend_thread.stop()
            self.stop_btn.setEnabled(False)
            print("3比特任务已手动停止")
            QMessageBox.information(self, "任务停止", "已停止当前3比特脉冲生成任务")

    def on_task_finish(self, success, msg):
        sys.stdout = self.original_stdout
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if success:
            print(f"3比特任务成功：{msg}")
            QMessageBox.information(self, "3比特任务成功", f"脉冲生成完成！\n{msg}")
        else:
            print(f"3比特任务失败：{msg}")
            QMessageBox.critical(self, "3比特任务失败", f"生成失败：\n{msg}")

# 3.2 2比特界面类
class Bit2PulseUI(QWidget):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.backend_thread = None
        self.original_stdout = sys.stdout  # 保存原始stdout
        self.log_stream = self.LogStream(self.log_signal)  # 自定义输出流
        self.init_ui()

    class LogStream(io.TextIOBase):
        def __init__(self, log_signal):
            super().__init__()
            self.log_signal = log_signal

        def write(self, text):
            if text.strip():
                self.log_signal.emit(text)

        def flush(self):
            pass

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 1. 标题
        title_label = QLabel("2比特量子计算机脉冲序列优化器")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 2. 参数输入区
        param_group = QGroupBox("2比特脉冲优化参数配置")
        param_group.setFont(QFont("微软雅黑", 11))
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(12)
        param_layout.setContentsMargins(15, 15, 15, 15)

        # 2.1 甲基峰频率
        param_layout.addWidget(QLabel("甲基左侧峰频率 (Hz)："), 0, 0)
        self.P1_edit = QDoubleSpinBox()
        self.P1_edit.setRange(-200, 200)
        self.P1_edit.setDecimals(2)
        self.P1_edit.setValue(-101.93)
        param_layout.addWidget(self.P1_edit, 0, 1)
        param_layout.addWidget(QLabel("甲基右侧峰频率 (Hz)："), 0, 2)
        self.P2_edit = QDoubleSpinBox()
        self.P2_edit.setRange(-200, 200)
        self.P2_edit.setDecimals(1)
        self.P2_edit.setValue(-90.0)
        param_layout.addWidget(self.P2_edit, 0, 3)

        # 2.2 通道频率
        param_layout.addWidget(QLabel("H通道频率 (MHz)："), 1, 0)
        self.wH_edit = QDoubleSpinBox()
        self.wH_edit.setRange(20, 30)
        self.wH_edit.setDecimals(6)
        self.wH_edit.setValue(27.029290)
        param_layout.addWidget(self.wH_edit, 1, 1)
        param_layout.addWidget(QLabel("P通道频率（*10^6） (MHz)："), 1, 2)
        self.wP_edit = QDoubleSpinBox()
        self.wP_edit.setRange(10, 12)
        self.wP_edit.setDecimals(6)
        self.wP_edit.setValue(10.941700)
        param_layout.addWidget(self.wP_edit, 1, 3)

        # 2.3 脉宽参数
        param_layout.addWidget(QLabel("H通道90°脉宽 (μs)："), 2, 0)
        self.PWH_edit = QDoubleSpinBox()
        self.PWH_edit.setRange(10, 100)
        self.PWH_edit.setDecimals(1)
        self.PWH_edit.setValue(40.0)
        param_layout.addWidget(self.PWH_edit, 2, 1)
        param_layout.addWidget(QLabel("P通道90°脉宽 (μs)："), 2, 2)
        self.PWP_edit = QDoubleSpinBox()
        self.PWP_edit.setRange(10, 100)
        self.PWP_edit.setDecimals(1)
        self.PWP_edit.setValue(40.0)
        param_layout.addWidget(self.PWP_edit, 2, 3)

        # 2.4 W参数
        param_layout.addWidget(QLabel("W_1 ："), 3, 0)
        self.w1_edit = QDoubleSpinBox()
        self.w1_edit.setRange(-100, 100)
        self.w1_edit.setDecimals(1)
        self.w1_edit.setValue(0.0)
        param_layout.addWidget(self.w1_edit, 3, 1)
        param_layout.addWidget(QLabel("W_2 ："), 3, 2)
        self.w2_edit = QDoubleSpinBox()
        self.w2_edit.setRange(-100, 100)
        self.w2_edit.setDecimals(1)
        self.w2_edit.setValue(0.0)
        param_layout.addWidget(self.w2_edit, 3, 3)
        param_layout.addWidget(QLabel("W_4 ："), 4, 0)
        self.w4_edit = QDoubleSpinBox()
        self.w4_edit.setRange(100, 300)
        self.w4_edit.setDecimals(1)
        self.w4_edit.setValue(170.0)
        param_layout.addWidget(self.w4_edit, 4, 1)

        # 2.5 温度参数
        param_layout.addWidget(QLabel("室内温度 (℃)："), 4, 2)
        self.T_edit = QDoubleSpinBox()
        self.T_edit.setRange(15, 35)
        self.T_edit.setDecimals(1)
        self.T_edit.setValue(24.0)
        param_layout.addWidget(self.T_edit, 4, 3)

        # 2.6 J耦合参数
        param_layout.addWidget(QLabel("耦合频率J12 (Hz)："), 5, 0)
        self.J12_edit = QDoubleSpinBox()
        self.J12_edit.setRange(600, 800)
        self.J12_edit.setDecimals(0)
        self.J12_edit.setValue(696)
        param_layout.addWidget(self.J12_edit, 5, 1)

        # 2.7 温漂参数
        param_layout.addWidget(QLabel("考虑温漂o1（逗号分隔）："), 5, 2)
        self.o1_edit = QLineEdit("-30,0,30")
        param_layout.addWidget(self.o1_edit, 5, 3)

        # 2.8 文件类型选择
        param_layout.addWidget(QLabel("输出文件类型："), 6, 0)
        self.file_type_group = QButtonGroup(self)
        self.spinq_radio = QRadioButton("spinq文件")
        self.spinq_radio.setChecked(True)
        self.json_radio = QRadioButton("json文件")
        self.file_type_group.addButton(self.spinq_radio, 1)
        self.file_type_group.addButton(self.json_radio, 0)
        file_type_layout = QHBoxLayout()
        file_type_layout.addWidget(self.spinq_radio)
        file_type_layout.addWidget(self.json_radio)
        param_layout.addLayout(file_type_layout, 6, 1)

        # 2.9 输出目录选择
        param_layout.addWidget(QLabel("脉冲输出目录："), 6, 2)
        self.output_dir_edit = QLineEdit(os.getcwd())
        self.output_dir_edit.setPlaceholderText("脉冲文件保存路径")
        param_layout.addWidget(self.output_dir_edit, 6, 3)
        self.output_dir_btn = QPushButton("选择目录")
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        param_layout.addWidget(self.output_dir_btn, 6, 4)

        main_layout.addWidget(param_group)

        # 3. 脉冲类型选择区
        pulse_group = QGroupBox("2比特脉冲类型选择（勾选需要优化的脉冲）")
        pulse_group.setFont(QFont("微软雅黑", 11))
        pulse_layout = QHBoxLayout(pulse_group)
        self.pulse_list = QListWidget()
        pulse_types = [
            "Gemini_Q1_X90", "Gemini_Q1_Y90", "Gemini_Q1_X90N", "Gemini_Q1_Y90N",
            "Gemini_Q2_X90", "Gemini_Q2_Y90", "Gemini_Q2_X90N", "Gemini_Q2_Y90N",
            "Gemini_Q1_X180", "Gemini_Q1_Y180", "Gemini_Q2_X180", "Gemini_Q2_Y180",
            "Gemini_Q1_H", "Gemini_Q2_H", "Gemini_CNOT12", "Gemini_CNOT21", 
            "Gemini_CZ", "Gemini_SWAP", "PPS_PART1", "PPS_PART2"
        ]
        for ptype in pulse_types:
            item = QListWidgetItem(ptype)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.pulse_list.addItem(item)
        pulse_layout.addWidget(self.pulse_list)
        btn_layout = QVBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_pulses)
        self.deselect_all_btn = QPushButton("取消全选")
        self.deselect_all_btn.clicked.connect(self.deselect_all_pulses)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        btn_layout.addStretch()
        pulse_layout.addLayout(btn_layout)
        main_layout.addWidget(pulse_group)

        log_group = QGroupBox("2比特运行日志（终端print信息）")
        log_group.setFont(QFont("微软雅黑", 11))
        log_layout = QVBoxLayout(log_group)

        self.log_textedit = QTextEdit()
        self.log_textedit.setReadOnly(True)
        self.log_textedit.setLineWrapMode(QTextEdit.NoWrap)
        self.log_textedit.setFont(QFont("Consolas", 10))
        self.log_textedit.setPlaceholderText("任务启动后，终端print信息将显示在这里...")

        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.clear_log)

        log_btn_layout = QHBoxLayout()
        log_btn_layout.addWidget(self.clear_log_btn)
        log_btn_layout.addStretch()
        log_layout.addLayout(log_btn_layout)
        log_layout.addWidget(self.log_textedit)

        main_layout.addWidget(log_group)

        # 4. 控制按钮区
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        self.start_btn = QPushButton("开始生成")
        self.start_btn.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_generation)
        self.stop_btn = QPushButton("停止任务")
        self.stop_btn.setFont(QFont("微软雅黑", 11, QFont.Bold))
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_generation)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        self.log_signal.connect(self.update_log)

    @pyqtSlot(str)
    def update_log(self, text):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_textedit.append(timestamp + text)
        self.log_textedit.moveCursor(self.log_textedit.textCursor().End)
        if self.log_textedit.document().blockCount() > 1000:
            self.log_textedit.clear()
            self.log_textedit.append("[日志清理] 已自动清理历史日志（超过1000行）")

    def clear_log(self):
        self.log_textedit.clear()
        self.log_textedit.append("[日志清空] 手动清理完成")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录", os.getcwd())
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def select_all_pulses(self):
        for i in range(self.pulse_list.count()):
            item = self.pulse_list.item(i)
            item.setCheckState(Qt.Checked)

    def deselect_all_pulses(self):
        for i in range(self.pulse_list.count()):
            item = self.pulse_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def get_selected_pulses(self):
        selected = []
        for i in range(self.pulse_list.count()):
            item = self.pulse_list.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected


    def start_generation(self):
        output_dir = self.output_dir_edit.text().strip()
        if not os.path.exists(output_dir):
            QMessageBox.warning(self, "目录不存在", "输出目录不存在，将自动创建！")
            os.makedirs(output_dir, exist_ok=True)

        selected_pulses = self.get_selected_pulses()
        if not selected_pulses:
            QMessageBox.warning(self, "未选择脉冲", "请至少勾选一种需要优化的脉冲类型！")
            return

        params = {
            "P1_methyl": self.P1_edit.value(),       # 甲基左侧峰频率（Hz）
            "P2_methyl": self.P2_edit.value(),       # 甲基右侧峰频率（Hz）
            "w_H": self.wH_edit.value(),             # H通道频率（MHz）
            "w_P": self.wP_edit.value(),             # P通道频率（MHz）
            "PW_H": self.PWH_edit.value(),           # H通道90°脉宽（μs）
            "PW_P": self.PWP_edit.value(),           # P通道90°脉宽（μs）
            "T_indoor": self.T_edit.value(),         # 室内温度（℃，后端转K）
            "w_1": self.w1_edit.value(),             # W1 参数
            "w_2": self.w2_edit.value(),             # W2 参数
            "w_4": self.w4_edit.value(),             # W4 参数
            "J12": self.J12_edit.value(),            # J12 耦合频率（Hz）
            "o1": self.o1_edit.text().strip(),       # 温漂参数（逗号分隔字符串）
            "File_Type": self.file_type_group.checkedId(),  # 文件类型（1=spinq，0=json）
            "output_dir": output_dir,                # 脉冲输出目录路径
            "selected_pulses": selected_pulses,      # 勾选的脉冲类型列表
            "start_time": time.time()                # 任务开始时间（用于耗时计算）
        }

        sys.stdout = self.log_stream
        print("2比特任务启动，开始初始化参数...")

        self.backend_thread = Bit2BackendThread(params)
        self.backend_thread.finish_signal.connect(self.on_task_finish)
        self.backend_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_generation(self):
        if self.backend_thread and self.backend_thread.isRunning():
            self.backend_thread.stop()
            self.stop_btn.setEnabled(False)
            print("2比特任务已手动停止")
            QMessageBox.information(self, "2比特任务停止", "已停止当前2比特脉冲优化任务")

    def on_task_finish(self, success, msg):
        sys.stdout = self.original_stdout

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if success:
            QMessageBox.information(self, "2比特任务成功", f"脉冲优化完成！\n{msg}")
        else:
            QMessageBox.critical(self, "2比特任务失败", f"优化失败：\n{msg}")

class CombinedPulseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("量子脉冲生成器（2比特+3比特+文件转换）")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        self.bit2_ui = Bit2PulseUI()
        self.bit3_ui = Bit3PulseUI()
        self.file_convert_ui = FileConverterUI() 
        self.tab_widget.addTab(self.bit2_ui, "2比特脉冲优化")
        self.tab_widget.addTab(self.bit3_ui, "3比特脉冲生成")
        self.tab_widget.addTab(self.file_convert_ui, "脉冲文件转换")

        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

class FileConverter:
    def __init__(self, file_path, output_path, convert_type, log_area):
        self.file_path = file_path
        self.output_path = output_path
        self.convert_type = convert_type
        self.log_area = log_area
        self.TITLE = ""
        self.TYPE = ""
        self.ORIGIN = ""
        self.OWNER = ""
        self.DATE = ""
        self.TIME = ""
        self.TOTALPULSEWIDTH = ""
        self.Calibration_Power = ""
        self.SLICES = ""
        self.PULSEWIDTH = ""
        self.pulse = {}
        self.read_file()
        
    def read_file(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
            pulse_data = []
            for line in lines:
                line = line.strip()
                if line.startswith('##') and not line.startswith('##END='):
                    parts = line[2:].split('=', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip(), parts[1].strip()
                        if key == 'TITLE': self.TITLE = value
                        elif key == 'TYPE': self.TYPE = value
                        elif key == 'ORIGIN': self.ORIGIN = value
                        elif key == 'OWNER': self.OWNER = value
                        elif key == 'DATE': self.DATE = value
                        elif key == 'TIME': self.TIME = value
                        elif key == 'TOTALPULSEWIDTH': self.TOTALPULSEWIDTH = float(value) * 1e3
                        elif key == 'Calibration_Power': self.Calibration_Power = float(value)
                        elif key == 'SLICES': self.SLICES = int(value)
                        elif key == 'PULSEWIDTH': self.PULSEWIDTH = float(value) * 1e6
                elif ',' in line and not line.startswith('##'):
                    values = line.split(',')
                    pulse_data.append([float(val.strip()) for val in values])
                elif line.startswith('##END='):
                    break
            self.pulse = self.convert_pulse_data(pulse_data, self.convert_type)
        except Exception as e:
            self.log_area.append(f"解析文件时出错: {str(e)}")
            self.pulse = {"channel1_pulse": []}
            
    def convert_pulse_data(self, pulse_data, convert_type):
        if not pulse_data:
            return {"channel1_pulse": []}
        json_data = {}
        if convert_type == "Gemini":
            channel1, channel2 = [], []
            for dp in pulse_data:
                qubit, w, p, a = dp
                obj = {"detuning": 0, "phase": p, "amplitude": a, "width": w}
                unobj = {"detuning": 0, "phase": 0, "amplitude": 0, "width": w}
                if qubit == 1:
                    channel1.append(obj), channel2.append(unobj)
                else:
                    channel2.append(obj), channel1.append(unobj)
            json_data = {"channel1_pulse": channel1, "channel2_pulse": channel2}
        elif convert_type == "Triangulum":
            channel1 = [{"detuning": 0, "phase": dp[1], "amplitude": dp[0], "width": dp[2]} for dp in pulse_data]
            json_data = {"channel1_pulse": channel1}
        return json_data

    def write_file(self):
        try:
            json_data = {
                "description": {
                    "TITLE": self.TITLE, "TYPE": self.TYPE, "ORIGIN": self.ORIGIN, "OWNER": self.OWNER,
                    "DATE": self.DATE, "TIME": self.TIME, "TOTALPULSEWIDTH": self.TOTALPULSEWIDTH,
                    "Calibration_Power": self.Calibration_Power, "SLICES": self.SLICES, "PULSEWIDTH": self.PULSEWIDTH
                },
                "pulse": self.pulse
            }
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
        except Exception as e:
            self.log_area.append(f"写入文件时出错: {str(e)}")

class FileConverterUI(QWidget):  # 从QMainWindow改为QWidget
    def __init__(self):
        super().__init__()
        self.input_folder = ""
        self.output_folder = ""
        self.convert_type = "Triangulum"
        self.init_ui()  # 统一方法名，与原有页面一致
        
    def init_ui(self):
        # 统一布局：边距20/20/20/20，间距15（与2比特/3比特页面一致）
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 统一标题样式：微软雅黑16号粗体，居中（与原有页面标题一致）
        title_label = QLabel("脉冲文件格式转换器")
        title_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 统一控件字体：微软雅黑11号（与原有页面参数区字体一致）
        default_font = QFont("微软雅黑", 11)

        # 1. 输入文件夹选择
        input_layout = QHBoxLayout()
        self.input_label = QLabel('输入文件夹：未选择')
        self.input_label.setFont(default_font)
        input_button = QPushButton('选择输入文件夹')
        input_button.setFont(default_font)
        input_button.clicked.connect(self.select_input_folder)
        input_layout.addWidget(self.input_label)
        input_layout.addWidget(input_button)
        main_layout.addLayout(input_layout)

        # 2. 输出文件夹选择
        output_layout = QHBoxLayout()
        self.output_label = QLabel('输出文件夹：未选择')
        self.output_label.setFont(default_font)
        output_button = QPushButton('选择输出文件夹')
        output_button.setFont(default_font)
        output_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(output_button)
        main_layout.addLayout(output_layout)

        # 3. 转换方法选择
        method_layout = QHBoxLayout()
        method_label = QLabel('转换方法:')
        method_label.setFont(default_font)
        self.method_combo = QComboBox()
        self.method_combo.setFont(default_font)
        self.method_combo.addItems(['Triangulum', 'Gemini'])
        self.method_combo.setCurrentIndex(0)
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.method_combo)
        main_layout.addLayout(method_layout)

        # 4. 转换按钮
        convert_button = QPushButton('开始转换')
        convert_button.setFont(default_font)
        convert_button.clicked.connect(self.start_conversion)
        convert_button.setMinimumHeight(50)
        main_layout.addWidget(convert_button)

        # 5. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 6. 日志区域
        log_label = QLabel('转换日志:')
        log_label.setFont(default_font)
        self.log_area = QTextEdit()
        self.log_area.setFont(default_font)
        self.log_area.setReadOnly(True)
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_area)

    # 以下功能逻辑完全保留，无修改
    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择输入文件夹')
        if folder:
            self.input_folder = folder
            self.input_label.setText(f'输入文件夹：{folder}')
            self.log_area.append(f'已选择输入文件夹: {folder}')
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '选择输出文件夹')
        if folder:
            self.output_folder = folder
            self.output_label.setText(f'输出文件夹：{folder}')
            self.log_area.append(f'已选择输出文件夹: {folder}')
        
    def on_method_changed(self):
        self.convert_type = self.method_combo.currentText()
    
    def start_conversion(self):
        if not self.input_folder or not self.output_folder:
            self.log_area.append('错误：请先选择输入和输出文件夹！')
            return
        self.log_area.append('开始转换文件...')
        spinq_files = []
        for root, _, files in os.walk(self.input_folder):
            for f in files:
                if f.endswith('.spinq'):
                    spinq_files.append(os.path.join(root, f))
        if not spinq_files:
            self.log_area.append('未找到任何.spinq文件！')
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(spinq_files))
        self.progress_bar.setValue(0)
        for i, file_path in enumerate(spinq_files):
            rel_path = os.path.relpath(file_path, self.input_folder)
            output_path = os.path.join(self.output_folder, rel_path)
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.log_area.append(f'处理文件: {rel_path}')
            self.convert_file(file_path, output_path, self.convert_type)
            self.progress_bar.setValue(i + 1)
        self.log_area.append('所有文件处理完成！')

    def convert_file(self, file_path, output_path, convert_type):
        try:
            file_converter = FileConverter(file_path, output_path, convert_type, self.log_area)
            file_converter.write_file()
            self.log_area.append(f'成功转换文件: {os.path.basename(file_path)}')
        except Exception as e:
            self.log_area.append(f'转换文件时出错: {str(e)}')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = CombinedPulseWindow()
    main_window.show()
    sys.exit(app.exec_())