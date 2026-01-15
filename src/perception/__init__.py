def __init__(self, model_path=None, device="cuda"):
        # --- 路径自动修正逻辑 START ---
        if model_path is None:
            # 动态获取项目根目录: 当前文件(vlm_interface.py) -> 上两级(src) -> 上三级(Neural-TAMP)
            current_file_path = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
            # 拼接正确的绝对路径
            model_path = os.path.join(project_root, "models", "Qwen2-VL-7B-Instruct")
        
        print(f"[VLM] Initializing Qwen2-VL from: {model_path}")
        # --- 路径自动修正逻辑 END ---

        self.device = device
        
        try:
            # 加载模型 (自动使用 bfloat16 节省显存)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("[VLM] Model loaded successfully.")
        except OSError:
            print(f"\n[VLM Critical Error] 无法找到模型文件。")
            print(f"请检查该路径下是否有 config.json: {model_path}")
            print("如果模型尚未下载，请先运行 download_qwen.py")
            raise
        except Exception as e:
            print(f"[VLM Error] Failed to load model: {e}")
            raise e