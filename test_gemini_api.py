# 导入必要库
import os
import google.generativeai as genai
from dotenv import load_dotenv


def test_gemini_connection():
    # 1. 加载.env文件中的API密钥
    load_dotenv()  # 从项目根目录的.env文件加载环境变量
    # 第6行
    api_key = os.getenv("GEMINI_API_KEY")  # 正确：参数为.env中的键名GEMINI_API_KEY
    genai.configure(api_key=api_key)
    # 2. 验证API密钥是否存在
    if not api_key:
        raise ValueError("未找到GEMINI_API_KEY，请检查.env文件是否配置正确")

    # 3. 配置Gemini API
    genai.configure(api_key=api_key)
    # 新增调试行（第9行）
    print(f"读取到的API密钥（前5位）：{api_key[:5] if api_key else '未读取到'}")

    # 4. 初始化Gemini模型（使用gemini-pro，适合文本测试）
    model = genai.GenerativeModel("gemini-2.5-pro")


    # 5. 发送简单测试请求（生成一段与项目相关的文本，验证功能）
    prompt = "请简要说明浮式风机平台的运动响应预测的重要性"
    print(f"测试请求：{prompt}")

    try:
        # 调用模型生成内容
        response = model.generate_content(prompt)

        # 6. 输出响应结果，验证是否成功
        print("\nGemini API响应：")
        print(response.text)
        print("\n测试成功：API调用正常，网络和密钥有效")

    except Exception as e:
        # 捕获并显示错误信息（用于排查问题）
        print(f"\n测试失败：{str(e)}")


if __name__ == "__main__":
    test_gemini_connection()