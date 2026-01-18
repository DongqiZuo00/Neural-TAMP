import os
import json
import openai
from typing import Dict, Any, Optional

class LLMInterface:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model
        
        if not self.api_key:
            print("⚠️ [LLM Interface] Warning: No API Key found.")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def predict(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        发送请求并返回解析后的 JSON 字典。
        包含基础的错误处理。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0, # 贪婪解码，保证逻辑稳定性
                response_format={"type": "json_object"} # 强制 JSON
            )
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from LLM")
                
            return json.loads(content)

        except json.JSONDecodeError:
            print("❌ [LLM Error] Failed to parse JSON response.")
            return {"error": "Invalid JSON", "plan": []}
        except Exception as e:
            print(f"❌ [LLM Error] API Call failed: {e}")
            return {"error": str(e), "plan": []}