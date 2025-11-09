import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 设置matplotlib字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from scipy.fft import rfft, rfftfreq
import logging
from typing import Tuple, Optional
import os
import pickle
from datetime import datetime


class FlowControlPredictor:
    """
    流量控制预测器
    使用流量数据和设定值来预测阀门开度变化
    """
    
    def __init__(
        self,
        data_file: str = 'flow_control_data.csv',
        large_window: int = 3000,
        small_window: int = 100,
        fft_features_count: int = 20,
        test_size: float = 0.2,
        random_state: int = 42,
        prediction_horizon: int = 1,  # 预测未来多少步
        model_save_path: str = 'flow_control_model.pkl'  # 模型保存路径
    ):
        """
        初始化预测器
        
        Args:
            data_file: 输入数据文件路径
            large_window: 大窗口大小
            small_window: 小窗口大小
            fft_features_count: FFT特征数量
            test_size: 测试集比例
            random_state: 随机种子
            prediction_horizon: 预测时间步长
            model_save_path: 模型保存路径
        """
        self.data_file = data_file
        self.large_window = large_window
        self.small_window = small_window
        self.fft_features_count = fft_features_count
        self.test_size = test_size
        self.random_state = random_state
        self.prediction_horizon = prediction_horizon
        self.model_save_path = model_save_path
        
        # 数据存储
        self.data: Optional[pd.DataFrame] = None
        self.model = LinearRegression()
        
        # 训练数据
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.test_predictions = None
        self.full_predictions = None
        
        # 训练配置信息
        self.training_config = {
            'large_window': large_window,
            'small_window': small_window,
            'fft_features_count': fft_features_count,
            'test_size': test_size,
            'random_state': random_state,
            'prediction_horizon': prediction_horizon
        }
        
        # 设置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> None:
        """加载CSV数据"""
        try:
            self.logger.info(f"正在加载数据: {self.data_file}")
            self.data = pd.read_csv(self.data_file)
            
            # 验证列数和列名
            if len(self.data.columns) < 3:
                raise ValueError(f"数据文件至少需要3列，当前有 {len(self.data.columns)} 列")
            
            # 重命名列以便使用
            column_mapping = {self.data.columns[0]: 'flow_real',    # 流量实时值
                              self.data.columns[1]: 'set_value',    # 设定值
                              self.data.columns[2]: 'valve_opening'} # 阀门开度值
            self.data = self.data.rename(columns=column_mapping)
            
            self.logger.info(f"数据加载完成，共 {len(self.data)} 行")
            self.logger.info(f"列名: {list(self.data.columns)}")
            
            # 显示数据统计信息
            self.logger.info(f"流量实时值范围: [{self.data['flow_real'].min():.3f}, {self.data['flow_real'].max():.3f}]")
            self.logger.info(f"设定值范围: [{self.data['set_value'].min():.3f}, {self.data['set_value'].max():.3f}]")
            self.logger.info(f"阀门开度范围: [{self.data['valve_opening'].min():.3f}, {self.data['valve_opening'].max():.3f}]")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据文件: {self.data_file}")
        except pd.errors.EmptyDataError:
            raise ValueError("数据文件为空")
        except Exception as e:
            raise ValueError(f"加载数据时出错: {str(e)}")
    
    def calculate_flow_error(self) -> np.ndarray:
        """
        计算流量误差（实时值与设定值的差值）
        
        Returns:
            流量误差数组
        """
        flow_error = self.data['flow_real'].values - self.data['set_value'].values
        return flow_error
    
    def calculate_flow_ratio(self) -> np.ndarray:
        """
        计算流量比例（实时值/设定值）
        
        Returns:
            流量比例数组
        """
        flow_ratio = self.data['flow_real'].values / (self.data['set_value'].values + 1e-8)  # 避免除零
        return flow_ratio
    
    def _calculate_fft_features(self, data: np.ndarray, window_size: int) -> np.ndarray:
        """
        计算FFT特征
        
        Args:
            data: 输入数据
            window_size: 窗口大小
            
        Returns:
            FFT特征数组
        """
        fft_result = rfft(data)
        fft_magnitude = np.abs(fft_result)
        
        # 提取指定数量的特征并归一化
        features = fft_magnitude[:self.fft_features_count]
        normalized_features = features / max(window_size, 1)
        
        return normalized_features
    
    def _extract_time_domain_features(self, 
                                    large_window_data: np.ndarray, 
                                    small_window_data: np.ndarray, 
                                    current_flow: float,
                                    current_set: float,
                                    current_valve: float) -> np.ndarray:
        """
        提取时域特征
        
        Args:
            large_window_data: 大窗口数据
            small_window_data: 小窗口数据
            current_flow: 当前流量值
            current_set: 当前设定值
            current_valve: 当前阀门开度
            
        Returns:
            时域特征数组
        """
        # 计算移动平均
        ma_large_flow = np.mean(large_window_data) if len(large_window_data) > 0 else 0
        ma_small_flow = np.mean(small_window_data) if len(small_window_data) > 0 else 0
        
        # 计算标准差
        std_large_flow = np.std(large_window_data) if len(large_window_data) > 1 else 0
        std_small_flow = np.std(small_window_data) if len(small_window_data) > 1 else 0
        
        # 计算当前状态相关特征
        flow_error = current_flow - current_set
        flow_ratio = current_flow / (current_set + 1e-8)  # 避免除零
        
        return np.array([
            current_flow, current_set, current_valve,
            ma_large_flow, ma_small_flow,
            std_large_flow, std_small_flow,
            flow_error, flow_ratio
        ])
    
    def extract_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取特征用于预测阀门开度变化
        
        Returns:
            特征矩阵X和目标向量y
        """
        self.logger.info(f"开始提取特征，大窗口: {self.large_window}, 小窗口: {self.small_window}")
        
        if self.data is None:
            raise ValueError("请先加载数据")
        
        flow_values = self.data['flow_real'].values
        set_values = self.data['set_value'].values
        valve_values = self.data['valve_opening'].values
        
        # 计算流量误差和比例
        flow_errors = self.calculate_flow_error()
        flow_ratios = self.calculate_flow_ratio()
        
        # 确定起始索引（需要足够数据用于窗口计算）
        start_idx = max(self.large_window, self.small_window)
        n_samples = len(flow_values) - start_idx - self.prediction_horizon  # 考虑预测时间步
        
        if n_samples <= 0:
            raise ValueError("数据不足，无法提取特征")
        
        features_list = []
        targets_list = []
        
        for i in range(n_samples):
            current_idx = start_idx + i
            
            # 提取流量数据的窗口
            large_flow_window = flow_values[current_idx - self.large_window + 1:current_idx + 1]
            small_flow_window = flow_values[current_idx - self.small_window + 1:current_idx + 1]
            
            # 提取误差数据的窗口
            large_error_window = flow_errors[current_idx - self.large_window + 1:current_idx + 1]
            small_error_window = flow_errors[current_idx - self.small_window + 1:current_idx + 1]
            
            # 计算流量FFT特征
            large_flow_fft = self._calculate_fft_features(large_flow_window, self.large_window)
            small_flow_fft = self._calculate_fft_features(small_flow_window, self.small_window)
            
            # 计算误差FFT特征
            large_error_fft = self._calculate_fft_features(large_error_window, self.large_window)
            small_error_fft = self._calculate_fft_features(small_error_window, self.small_window)
            
            # 组合FFT特征
            combined_fft_features = np.concatenate([
                large_flow_fft, small_flow_fft,
                large_error_fft, small_error_fft
            ])
            
            # 计算时域特征
            time_features = self._extract_time_domain_features(
                large_flow_window, small_flow_window,
                flow_values[current_idx], set_values[current_idx], valve_values[current_idx]
            )
            
            # 计算基于设定值的额外特征
            set_fft_features = self._calculate_fft_features(
                set_values[current_idx - self.small_window + 1:current_idx + 1], 
                self.small_window
            )
            
            # 组合所有特征
            all_features = np.concatenate([
                combined_fft_features, 
                time_features, 
                set_fft_features[:self.fft_features_count//2]  # 减少设定值特征数量
            ])
            
            # 处理NaN值
            all_features = np.nan_to_num(all_features)
            
            features_list.append(all_features)
            
            # 目标：预测未来阀门开度变化
            target_idx = current_idx + self.prediction_horizon
            if target_idx < len(valve_values):
                target_valve = valve_values[target_idx]
                current_valve = valve_values[current_idx]
                valve_change = target_valve - current_valve  # 阀门开度变化量
                targets_list.append(valve_change)
            else:
                break
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        self.logger.info(f"特征提取完成，特征维度: {X.shape}, 目标维度: {y.shape}")
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> 'FlowControlPredictor':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 目标向量（阀门开度变化量）
            
        Returns:
            自身实例（支持链式调用）
        """
        self.logger.info("开始训练模型...")
        
        # 分割训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        self.logger.info(f"训练集大小: {len(self.X_train)}, 测试集大小: {len(self.X_test)}")
        
        # 训练模型
        self.model.fit(self.X_train, self.y_train)
        
        # 预测测试集
        self.test_predictions = self.model.predict(self.X_test)
        
        # 评估模型
        self._evaluate_model()
        
        # 预测全量数据
        self.full_predictions = self.model.predict(X)
        
        self.logger.info("模型训练完成")
        return self
    
    def _evaluate_model(self) -> None:
        """评估模型性能"""
        # 训练集评估
        y_train_pred = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        # 测试集评估
        test_mse = mean_squared_error(self.y_test, self.test_predictions)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, self.test_predictions)
        
        # 记录评估结果
        evaluation_results = {
            'train_mse': train_mse,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
        
        self.logger.info("模型性能评估:")
        for metric, value in evaluation_results.items():
            self.logger.info(f"{metric.upper()}: {value:.6f}")
        
        # 绘制预测对比图
        self._plot_predictions()
    
    def _plot_predictions(self) -> None:
        """绘制预测结果对比图"""
        plt.figure(figsize=(12, 6))
        
        # 绘制部分测试数据对比
        n_plot = min(500, len(self.y_test))
        plt.scatter(range(n_plot), self.y_test[:n_plot], 
                   label='实际阀门开度变化', alpha=0.6, s=20)
        plt.scatter(range(n_plot), self.test_predictions[:n_plot], 
                   label='预测阀门开度变化', alpha=0.6, s=20)
        
        plt.xlabel('样本索引')
        plt.ylabel('阀门开度变化')
        plt.title('实际值与预测值对比（部分测试数据）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = 'valve_change_prediction_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"预测对比图已保存至: {output_file}")
    
    def save_model(self) -> None:
        """
        保存训练好的模型为pkl文件
        """
        model_data = {
            'model': self.model,
            'training_config': self.training_config,
            'feature_names': self._get_feature_names(),
            'model_type': 'FlowControlPredictor',
            'created_at': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"模型已保存至: {self.model_save_path}")
    
    def _get_feature_names(self) -> list:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        feature_names = []
        
        # FFT特征名称
        for i in range(self.fft_features_count * 4):  # 4种FFT特征
            feature_names.append(f'fft_feature_{i}')
        
        # 时域特征名称
        time_feature_names = [
            'current_flow', 'current_set', 'current_valve',
            'ma_large_flow', 'ma_small_flow',
            'std_large_flow', 'std_small_flow',
            'flow_error', 'flow_ratio'
        ]
        feature_names.extend(time_feature_names)
        
        # 设定值FFT特征名称
        for i in range(self.fft_features_count // 2):
            feature_names.append(f'set_fft_feature_{i}')
        
        return feature_names
    
    def generate_predictions(self) -> pd.DataFrame:
        """
        生成完整预测结果
        
        Returns:
            包含预测结果的数据框
        """
        if self.data is None or self.full_predictions is None:
            raise ValueError("请先完成数据加载和模型训练")
        
        self.logger.info("开始生成完整预测结果...")
        
        # 初始化预测数组
        predicted_valve_changes = np.full(len(self.data), np.nan)
        predicted_valve_positions = np.full(len(self.data), np.nan)
        
        # 填充预测结果（从窗口开始位置之后）
        start_idx = max(self.large_window, self.small_window)
        for i, pred_change in enumerate(self.full_predictions):
            pred_idx = start_idx + i + self.prediction_horizon
            if pred_idx < len(predicted_valve_changes):
                predicted_valve_changes[pred_idx] = pred_change
                # 计算预测的阀门位置
                current_valve = self.data['valve_opening'].iloc[pred_idx - self.prediction_horizon]
                predicted_valve_positions[pred_idx] = current_valve + pred_change
        
        # 创建结果数据框
        result_df = self.data.copy()
        result_df['predicted_valve_change'] = predicted_valve_changes
        result_df['predicted_valve_position'] = predicted_valve_positions
        
        # 限制阀门开度范围 [0, 1]
        result_df['predicted_valve_position'] = np.clip(
            result_df['predicted_valve_position'], 0.0, 1.0
        )
        
        # 计算流量误差和比例
        result_df['flow_error'] = self.calculate_flow_error()
        result_df['flow_ratio'] = self.calculate_flow_ratio()
        
        # 保存结果
        output_file = 'flow_control_predictions.csv'
        result_df.to_csv(output_file, index=False)
        self.logger.info(f"预测结果已保存至: {output_file}")
        
        # 统计信息
        valid_predictions = predicted_valve_changes[~np.isnan(predicted_valve_changes)]
        if len(valid_predictions) > 0:
            self.logger.info(f"有效预测数量: {len(valid_predictions)}")
            self.logger.info(f"预测阀门开度变化范围: [{valid_predictions.min():.6f}, {valid_predictions.max():.6f}]")
            self.logger.info(f"预测阀门开度变化均值: {valid_predictions.mean():.6f}")
            self.logger.info(f"预测阀门开度变化标准差: {valid_predictions.std():.6f}")
        
        # 绘制结果对比图
        self._plot_valve_comparison(result_df)
        
        # 保存模型
        self.save_model()
        
        return result_df
    
    def _plot_valve_comparison(self, result_df: pd.DataFrame) -> None:
        """绘制阀门开度对比图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # 采样数据以提高绘图性能
        sample_rate = max(1, len(result_df) // 1000)  # 最多1000个点
        sampled_indices = range(0, len(result_df), sample_rate)
        sampled_df = result_df.iloc[sampled_indices]
        
        # 上图：阀门开度对比
        ax1.plot(sampled_df.index, sampled_df['valve_opening'], 
                label='实际阀门开度', alpha=0.7, linewidth=1.5)
        ax1.plot(sampled_df.index, sampled_df['predicted_valve_position'], 
                label='预测阀门开度', alpha=0.7, linewidth=1.5)
        
        ax1.set_xlabel('数据索引')
        ax1.set_ylabel('阀门开度')
        ax1.set_title('实际阀门开度 vs 预测阀门开度')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标记预测开始位置
        start_idx = max(self.large_window, self.small_window) + self.prediction_horizon
        ax1.axvline(x=start_idx, color='red', linestyle='--', 
                   label=f'预测开始位置', alpha=0.8)
        
        # 下图：流量误差
        ax2.plot(sampled_df.index, sampled_df['flow_error'], 
                label='流量误差 (实时值-设定值)', alpha=0.7, linewidth=1.5)
        ax2.set_xlabel('数据索引')
        ax2.set_ylabel('流量误差')
        ax2.set_title('流量误差变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = 'flow_control_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"分析图表已保存至: {output_file}")
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        运行完整管道
        
        Returns:
            包含预测结果的数据框
        """
        self.logger.info("开始运行完整预测管道...")
        
        # 加载数据
        self.load_data()
        
        # 提取特征
        X, y = self.extract_features()
        
        # 训练模型
        self.train_model(X, y)
        
        # 生成预测
        result = self.generate_predictions()
        
        self.logger.info("完整管道运行完成")
        return result


def load_model(model_path: str) -> dict:
    """
    加载pkl模型文件
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        包含模型和其他信息的字典
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_with_loaded_model(model_path: str, new_features: np.ndarray) -> np.ndarray:
    """
    使用加载的模型进行预测
    
    Args:
        model_path: 模型文件路径
        new_features: 新的特征数据
        
    Returns:
        预测结果
    """
    model_data = load_model(model_path)
    model = model_data['model']
    predictions = model.predict(new_features)
    return predictions


def main():
    """主函数"""
    # 配置参数
    config = {
        'data_file': 'flow_control_data.csv',  # 你的CSV文件名
        'large_window': 3000,
        'small_window': 100,
        'fft_features_count': 20,
        'test_size': 0.2,
        'random_state': 42,
        'prediction_horizon': 1,  # 预测下一个时间步的阀门变化
        'model_save_path': 'flow_control_model.pkl'  # 模型保存路径
    }
    
    # 创建预测器实例
    predictor = FlowControlPredictor(**config)
    
    try:
        # 运行完整管道
        result = predictor.run_pipeline()
        print("流量控制预测管道执行成功！")
        print(f"预测结果已保存至: flow_control_predictions.csv")
        print(f"训练模型已保存至: flow_control_model.pkl")
        
        # 验证模型保存
        print("\n验证模型加载功能...")
        loaded_model_data = load_model(config['model_save_path'])
        print(f"加载的模型类型: {loaded_model_data['model_type']}")
        print(f"模型创建时间: {loaded_model_data['created_at']}")
        print(f"特征数量: {len(loaded_model_data['feature_names'])}")
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()