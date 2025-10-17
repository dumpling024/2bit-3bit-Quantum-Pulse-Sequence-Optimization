import os
import json
from datetime import datetime
import numpy as np

def write_pulse_file(B_np, M, t_pulse, outputfile, fidelity, local_vars):
    """将脉冲序列写入.spinq文件（纯NumPy实现，修复detach错误）"""
    try:
        total_time = M * t_pulse
        # 关键修改：local_vars.calib已是NumPy数组，无需detach转换
        calib_np = local_vars.calib  # 直接使用NumPy数组
        complex_pulse = B_np[:M] + 1j * B_np[M:]  # NumPy复数运算
        amp = np.abs(complex_pulse) * 100 / np.abs(calib_np)
        phase = np.mod((180 / np.pi) * np.angle(complex_pulse), 360)

        target_pulse_idx = local_vars.pulsenumber
        
        # 确保B_store是字典类型（初始化如果不存在）
        if not hasattr(local_vars, 'B_store'):
            local_vars.B_store = {}
        elif not isinstance(local_vars.B_store, dict):
            raise TypeError("B_store必须是字典类型")
        
        # 存储当前B值（B_np已是NumPy数组，无需转换）
        local_vars.B_store[target_pulse_idx] = np.asarray(B_np)  # 确保为NumPy格式
        
        # 验证B的长度是否为2*M
        if len(B_np) != 2 * M:
            raise ValueError(f"B的长度必须为2*M ({2*M})，实际为{len(B_np)}")
        
        # 构建脉冲矩阵：第一列是X分量，第二列是Y分量
        pulse = np.zeros((M, 2), dtype=np.float64)
        for kk in range(M):
            pulse[kk, 0] = B_np[kk]          # X分量（前M个元素）
            pulse[kk, 1] = B_np[kk + M]      # Y分量（后M个元素）
        
        # 写入文件
        with open(outputfile, 'w', encoding='utf-8') as shpfile:
            # 写入头部信息
            shpfile.write(f"##TITLE= {outputfile}\n")
            shpfile.write("##DATA TYPE= Shape Data\n")
            shpfile.write("##ORIGIN= SpinQ GRAPE Pulses \n")
            shpfile.write(f"##OWNER= {local_vars.user}\n")
            shpfile.write(f"##DATE= {datetime.now().strftime('%Y-%m-%d')}\n")
            
            # 当前时间
            current_time = datetime.now()
            shpfile.write(f"##GATE_FIDELITY= {fidelity:.6f}\n")
            shpfile.write(f"##TIME= {current_time.hour}:{current_time.minute}\n")
            
            # 写入极值信息
            shpfile.write(f"##MINX= {np.min(amp):7.6e}\n")
            shpfile.write(f"##MAXX= {np.max(amp):7.6e}\n")
            shpfile.write(f"##MINY= {np.min(phase):7.6e}\n")
            shpfile.write(f"##MAXY= {np.max(phase):7.6e}\n")
            
            # 写入脉冲时间信息
            shpfile.write(f"##TOTALPULSEWIDTH= {total_time}\n")
            # 关键修改：直接使用NumPy数组的标量值
            shpfile.write(f"##Calibration_Power= {calib_np.item()}\n")  # .item()获取标量
            shpfile.write(f"##SLICES= {len(amp)}\n")
            shpfile.write(f"##PULSEWIDTH= {t_pulse}\n")
            
            # 写入脉冲数据
            for ct in range(len(amp)):
                time_us = t_pulse * 1e6
                shpfile.write(f"  {amp[ct]:7.6e},  {phase[ct]:7.6e},  {time_us:7.6e}\n")
            
            # 脉冲编号35时添加额外行
            if target_pulse_idx == 35:
                shpfile.write(f"  {0.0:7.6e},  {0.0:7.6e},  {2e5:7.6e}\n")
            
            shpfile.write("##END=\n")
        
        print(f"脉冲数据已保存到 {outputfile}")
        return True

    except Exception as e:
        print(f"写入脉冲文件错误: {str(e)}")
        return False

def write_pulse_file_json(B_np, M, t_pulse, outputfile, fidelity, local_vars):
    """将脉冲序列写入JSON文件（纯NumPy实现，修复detach错误）"""
    try:
        # 处理文件名
        filename, ext = os.path.splitext(outputfile)
        if ext.lower() == '.spinq':
            outputfile = f"{filename}.json"
        else:
            outputfile = f"{outputfile}.json"

        # 基础参数计算
        total_time = M * t_pulse
        # 关键修改：移除detach，直接处理NumPy数组（.item()转标量）
        calib = local_vars.calib.item() if isinstance(local_vars.calib, np.ndarray) else float(local_vars.calib)
        complex_pulse = B_np[:M] + 1j * B_np[M:]
        amp = np.abs(complex_pulse) * 100 / np.abs(calib)
        phase = np.mod((180 / np.pi) * np.angle(complex_pulse), 360)
        target_pulse_idx = local_vars.pulsenumber

        # B_store存储逻辑
        if not hasattr(local_vars, 'B_store'):
            local_vars.B_store = {}
        elif not isinstance(local_vars.B_store, dict):
            raise TypeError("B_store必须是字典类型")
        # B_np已是NumPy数组，直接存储
        local_vars.B_store[target_pulse_idx] = np.asarray(B_np)

        # 参数合法性验证
        if len(B_np) != 2 * M:
            raise ValueError(f"B的长度必须为2*M ({2*M})，实际为{len(B_np)}")
        if not hasattr(local_vars, 'user'):
            raise AttributeError("local_vars中缺少'user'属性（用于JSON的OWNER字段）")

        # 构建JSON数据结构
        description = {
            "TITLE": outputfile.split("\\")[-1] if "\\" in outputfile else outputfile,
            "TYPE": "",
            "ORIGIN": "SpinQ GRAPE Pulses",
            "OWNER": local_vars.user,
            "DATE": datetime.now().strftime("%d-%b-%Y"),
            "TIME": datetime.now().strftime("%H:%M"),
            "TOTALPULSEWIDTH": round(total_time * 1e6, 1),
            "Calibration_Power": round(calib, 3),
            "SLICES": len(amp),
            "PULSEWIDTH": round(t_pulse * 1e6, 1)
        }

        channel1_pulse = []
        pulse_width_us = round(t_pulse * 1e6, 1)
        for idx in range(len(amp)):
            pulse_slice = {
                "detuning": 0,
                "phase": round(phase[idx], 4),
                "amplitude": round(amp[idx], 2),
                "width": pulse_width_us
            }
            channel1_pulse.append(pulse_slice)

        # 脉冲编号35时添加额外行
        if target_pulse_idx == 35:
            extra_slice = {
                "detuning": 0,
                "phase": 0.0,
                "amplitude": 0.0,
                "width": pulse_width_us
            }
            channel1_pulse.append(extra_slice)

        json_data = {
            "description": description,
            "pulse": {
                "channel1_pulse": channel1_pulse
            }
        }

        # 写入JSON文件
        with open(outputfile, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=2, ensure_ascii=False)

        print(f"脉冲JSON文件已保存到 {outputfile}")
        return True

    except Exception as e:
        print(f"写入脉冲JSON文件错误: {str(e)}")
        return False
    
def save_fit_result_to_file(result, filename="fit_params.json"):
    """将拟合结果保存到JSON文件（处理NumPy类型转换）"""
    # 递归转换NumPy类型为Python原生类型（避免JSON序列化错误）
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()  # NumPy标量转Python标量
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPy数组转Python列表
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}  # 字典递归转换
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]  # 列表/元组递归转换
        else:
            return obj  # 其他类型保持不变
    
    # 转换结果类型
    converted_result = convert_numpy_types(result)
    
    # 写入文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(converted_result, f, ensure_ascii=False, indent=4)  # indent=4增强可读性
    print(f"拟合结果已保存到: {filename}")

def dataout_SpinQ(file_path, operation, param_24, M):
    """
    适配包含头部注释的 .spinq 格式：
    - 头部：##开头的注释行（需跳过）
    - 数据：每行3列数值（第1列时间，第2列幅度，第3列忽略）
    - 结尾：##END=标记（需跳过）
    
    参数与返回值同前，确保与 MATLAB 逻辑完全对齐
    """
    # 1. 读取文件所有行
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 2. 过滤行：跳过##注释行、空行、终止标记
    valid_data_lines = []
    for line in lines:
        line_stripped = line.strip()  # 去除首尾空格和换行符
        
        # 跳过空行
        if not line_stripped:
            continue
        
        # 跳过##开头的注释行（包括头部注释和结尾##END=）
        if line_stripped.startswith('##'):
            continue
        
        # 剩余的是有效数据行
        valid_data_lines.append(line_stripped)
    
    # 3. 检查有效数据行数是否满足需求（至少M行）
    if len(valid_data_lines) < M:
        raise ValueError(
            f"数据行不足！文件中有效数据行：{len(valid_data_lines)} 行，"
            f"需要读取 {M} 行（M为脉冲长度）"
        )
    
    # 4. 解析前M行数据（第1列时间，第2列幅度）
    t_list = []
    amp_list = []
    for i in range(M):
        data_line = valid_data_lines[i]
        # 按空格分割（支持多个空格，适配文件中"  0.000000e+00,  1.800000e+02, ..."格式）
        # 先替换逗号为空格，再分割（处理", "分隔符）
        nums_str = data_line.replace(',', ' ').split()
        # 转换为浮点数（确保每行有3个数值）
        try:
            nums = [float(num) for num in nums_str]
        except ValueError:
            raise ValueError(f"第 {i+1} 行数据格式错误，无法转换为数值：{data_line}")
        
        if len(nums) != 3:
            raise ValueError(
                f"第 {i+1} 行数据列数错误！需要3列，实际有 {len(nums)} 列：{data_line}"
            )
        
        t_list.append(nums[0])    # 第1列：时间
        amp_list.append(nums[1])  # 第2列：幅度
    
    # 5. 处理相位（同前：若文件无相位数据，默认返回0；有则替换）
    phase = np.zeros(M, dtype=np.float64)  # 可根据实际场景修改（如从其他列读取）
    
    # 6. 转换为numpy数组（与MATLAB输出格式一致）
    t_gatetype1 = np.array(t_list, dtype=np.float64)
    amp = np.array(amp_list, dtype=np.float64)
    
    return amp, phase, t_gatetype1