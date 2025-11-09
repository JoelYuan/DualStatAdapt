使用方法：
将你的CSV文件命名为 flow_control_data.csv 或修改配置中的文件名
确保CSV文件的前三列分别是：流量实时值、设定值、阀门开度值
运行[训练自适应控制.py]脚本即可获得预测结果

解释一下三个数据列（流量实时值、设定值、阀门开度值）是如何构造成A和B的：

## 数据结构说明
- **第1列**：流量实时值 (flow_real)
- **第2列**：设定值 (set_value) 
- **第3列**：阀门开度值 (valve_opening)

## 构造A（特征）的步骤

### 1. **计算流量误差**
```python
flow_error = flow_real - set_value
```
- 表示当前流量与目标设定的偏差

### 2. **计算流量比例**
```python
flow_ratio = flow_real / set_value
```
- 表示当前流量相对于设定值的比例

### 3. **提取滑动窗口数据**
- **大窗口**（3000个点）：从历史数据中提取较长时期的趋势
- **小窗口**（100个点）：从近期数据中提取快速变化

### 4. **计算FFT特征**
```python
# 对流量数据进行FFT变换
large_flow_fft = FFT(flow_real_window_3000)
small_flow_fft = FFT(flow_real_window_100)

# 对流量误差数据进行FFT变换  
large_error_fft = FFT(flow_error_window_3000)
small_error_fft = FFT(flow_error_window_100)

# 对设定值数据进行FFT变换
set_fft = FFT(set_value_window_100)
```

### 5. **提取时域特征**
```python
# 当前状态特征
current_features = [
    flow_real_current,      # 当前流量
    set_value_current,      # 当前设定值
    valve_opening_current,  # 当前阀门开度
    flow_error_current,     # 当前流量误差
    flow_ratio_current      # 当前流量比例
]

# 统计特征
statistical_features = [
    mean_flow_3000,         # 3000点流量均值
    mean_flow_100,          # 100点流量均值
    std_flow_3000,          # 3000点流量标准差
    std_flow_100,           # 100点流量标准差
]
```

### 6. **组合所有特征（构成A）**
```python
A = [
    # FFT特征（频域）
    large_flow_fft_features,    # 大窗口流量FFT特征
    small_flow_fft_features,    # 小窗口流量FFT特征
    large_error_fft_features,   # 大窗口误差FFT特征
    small_error_fft_features,   # 小窗口误差FFT特征
    set_fft_features,           # 设定值FFT特征
    
    # 时域特征
    current_features,           # 当前状态特征
    statistical_features        # 统计特征
]
```

## 构造B（目标）的步骤

### 1. **定义预测目标**
```python
# B = 阀门开度的变化量
valve_change = valve_opening_future - valve_opening_current
```

### 2. **时间对齐**
```python
# 当前特征时间点: t
# 目标时间点: t + prediction_horizon (通常是t+1)

A[t] -> B[t + 1]  # 用t时刻的特征预测t+1时刻的阀门变化
```

## 具体构造过程示例

假设我们有以下数据：
```
时间点 | 流量实时值 | 设定值 | 阀门开度
   1   |    100    |  105   |   0.6
   2   |    102    |  105   |   0.62
   3   |    104    |  105   |   0.63
   4   |    103    |  105   |   0.61
   5   |    101    |  105   |   0.59
```

### 构造A[3]（时间点3的特征）：
- 使用时间点1-3的流量数据计算大窗口特征
- 使用时间点2-3的流量数据计算小窗口特征  
- 计算时间点3的流量误差：104 - 105 = -1
- 计算时间点3的流量比例：104 / 105 = 0.99
- 组合所有特征形成A[3]

### 构造B[3]（时间点3的目标）：
- B[3] = 阀门开度[4] - 阀门开度[3] = 0.61 - 0.63 = -0.02

## 最终关系
```
A[t] (当前特征) -> B[t+1] (下一时刻的阀门开度变化)
```

这样模型就学会了根据当前的流量状态（实时值、设定值、历史趋势）来预测下一步应该调整多少阀门开度。
