import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional

class SummaryVisualizer:
    """摘要可视化工具，用于生成模型性能对比图表"""
    
    def __init__(self, output_dir: str = "visualization_results"):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出结果目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_rouge_scores(
        self, 
        model_results: Dict[str, Dict[str, Any]], 
        output_path: Optional[str] = None
    ):
        """
        绘制Rouge指标对比图
        
        Args:
            model_results: 模型评估结果字典，格式为 {model_name: results}
            output_path: 输出文件路径，默认为 output_dir/rouge_comparison.png
        """
        if not model_results:
            print("没有模型结果可供可视化")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "rouge_comparison.png")
        
        model_names = list(model_results.keys())
        rouge_1_f = [results['rouge_scores']['rouge-1']['f'] for results in model_results.values()]
        rouge_2_f = [results['rouge_scores']['rouge-2']['f'] for results in model_results.values()]
        rouge_l_f = [results['rouge_scores']['rouge-l']['f'] for results in model_results.values()]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, rouge_1_f, width, label='Rouge-1-F1')
        plt.bar(x, rouge_2_f, width, label='Rouge-2-F1')
        plt.bar(x + width, rouge_l_f, width, label='Rouge-L-F1')
        
        plt.xlabel('模型')
        plt.ylabel('Rouge F1分数')
        plt.title('不同模型的Rouge指标对比')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Rouge指标对比图已保存到 {output_path}")
    
    def plot_sentiment_consistency(
        self, 
        model_results: Dict[str, Dict[str, Any]], 
        output_path: Optional[str] = None
    ):
        """
        绘制情感一致性对比图
        
        Args:
            model_results: 模型评估结果字典，格式为 {model_name: results}
            output_path: 输出文件路径，默认为 output_dir/sentiment_consistency.png
        """
        if not model_results:
            print("没有模型结果可供可视化")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "sentiment_consistency.png")
        
        model_names = list(model_results.keys())
        label_consistency = [results['sentiment_consistency']['label_consistency'] for results in model_results.values()]
        score_consistency = [results['sentiment_consistency']['score_consistency'] for results in model_results.values()]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, label_consistency, width, label='标签一致性')
        plt.bar(x + width/2, score_consistency, width, label='分数一致性')
        
        plt.xlabel('模型')
        plt.ylabel('情感一致性分数')
        plt.title('不同模型的情感一致性对比')
        plt.xticks(x, model_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"情感一致性对比图已保存到 {output_path}")
    
    def create_metrics_heatmap(
        self, 
        model_results: Dict[str, Dict[str, Any]], 
        output_path: Optional[str] = None
    ):
        """
        创建性能指标热力图
        
        Args:
            model_results: 模型评估结果字典，格式为 {model_name: results}
            output_path: 输出文件路径，默认为 output_dir/metrics_heatmap.png
        """
        if not model_results:
            print("没有模型结果可供可视化")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "metrics_heatmap.png")
        
        model_names = list(model_results.keys())
        metrics = {
            'Rouge-1-F1': [results['rouge_scores']['rouge-1']['f'] for results in model_results.values()],
            'Rouge-2-F1': [results['rouge_scores']['rouge-2']['f'] for results in model_results.values()],
            'Rouge-L-F1': [results['rouge_scores']['rouge-l']['f'] for results in model_results.values()],
            '情感标签一致性': [results['sentiment_consistency']['label_consistency'] for results in model_results.values()],
            '情感分数一致性': [results['sentiment_consistency']['score_consistency'] for results in model_results.values()]
        }
        
        df = pd.DataFrame(metrics, index=model_names)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('模型性能指标热力图')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"性能指标热力图已保存到 {output_path}")
    
    def create_comparison_dashboard(
        self, 
        model_results: Dict[str, Dict[str, Any]], 
        output_path: Optional[str] = None
    ):
        """
        创建模型对比仪表板（HTML格式）
        
        Args:
            model_results: 模型评估结果字典，格式为 {model_name: results}
            output_path: 输出文件路径，默认为 output_dir/comparison_dashboard.html
        """
        if not model_results:
            print("没有模型结果可供可视化")
            return
        
        if output_path is None:
            output_path = os.path.join(self.output_dir, "comparison_dashboard.html")
        
        rouge_path = os.path.join(self.output_dir, "rouge_comparison.png")
        self.plot_rouge_scores(model_results, rouge_path)
        
        sentiment_path = os.path.join(self.output_dir, "sentiment_consistency.png")
        self.plot_sentiment_consistency(model_results, sentiment_path)
        
        heatmap_path = os.path.join(self.output_dir, "metrics_heatmap.png")
        self.create_metrics_heatmap(model_results, heatmap_path)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>中文新闻摘要模型对比仪表板</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{
                    color: #333;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>中文新闻摘要模型对比仪表板</h1>
                
                <h2>Rouge指标对比</h2>
                <div class="chart-container">
                    <img src="rouge_comparison.png" alt="Rouge指标对比">
                </div>
                
                <h2>情感一致性对比</h2>
                <div class="chart-container">
                    <img src="sentiment_consistency.png" alt="情感一致性对比">
                </div>
                
                <h2>性能指标热力图</h2>
                <div class="chart-container">
                    <img src="metrics_heatmap.png" alt="性能指标热力图">
                </div>
                
                <h2>模型性能数据表</h2>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>Rouge-1-F1</th>
                        <th>Rouge-2-F1</th>
                        <th>Rouge-L-F1</th>
                        <th>情感标签一致性</th>
                        <th>情感分数一致性</th>
                    </tr>
        """
        
        for model_name, results in model_results.items():
            html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{results['rouge_scores']['rouge-1']['f']:.3f}</td>
                        <td>{results['rouge_scores']['rouge-2']['f']:.3f}</td>
                        <td>{results['rouge_scores']['rouge-l']['f']:.3f}</td>
                        <td>{results['sentiment_consistency']['label_consistency']:.3f}</td>
                        <td>{results['sentiment_consistency']['score_consistency']:.3f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
                
                <h2>生成时间</h2>
                <p>生成时间：<span id="timestamp"></span></p>
                
                <script>
                    document.getElementById('timestamp').textContent = new Date().toLocaleString();
                </script>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"模型对比仪表板已保存到 {output_path}")
