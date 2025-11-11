import os
import google.generativeai as genai
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class GeminiService:
    def __init__(self):
        # 配置API密钥
        genai.configure(api_key=os.getenv("AIzaSyCs6pw8_ujt0ZlwetsH8qzdj1xtW1Ql3FI"))
        # 初始化模型
        self.model = genai.GenerativeModel('gemini 2.5 pro')
        self.chat = None

    def analyze_prediction_results(self, metrics, model_name):
        """利用Gemini分析预测结果"""
        prompt = f"""
        作为海洋工程AI助手，请分析以下浮式风机平台运动响应预测模型的性能指标：
        模型名称: {model_name}
        性能指标: {metrics}

        请提供以下分析：
        1. 模型性能评估（优势与不足）
        2. 可能的优化方向
        3. 对实际海洋工程应用的建议
        """

        response = self.model.generate_content(prompt)
        return response.text

    def generate_report_summary(self, experiment_results):
        """生成实验报告摘要"""
        prompt = f"""
        请总结以下浮式风机平台运动响应预测实验结果：
        {experiment_results}

        总结应包括：
        1. 各模型性能对比
        2. 最佳策略分析
        3. 实际应用价值
        请使用专业但简洁的语言，适合纳入工程报告。
        """

        response = self.model.generate_content(prompt)
        return response.text

    def start_chat_session(self):
        """开始聊天会话"""
        self.chat = self.model.start_chat(history=[])
        return self.chat

    def chat_with_model(self, message):
        """与模型进行对话"""
        if not self.chat:
            self.start_chat_session()
        response = self.chat.send_message(message)
        return response.text