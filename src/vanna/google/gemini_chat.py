import os
from ..base import VannaBase


class GoogleGeminiChat(VannaBase):
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)

        # default temperature - can be overrided using config
        self.temperature = 1.0

        if "temperature" in config:
            self.temperature = config["temperature"]

        if "LLM_model_name" in config:
            model_name = config["LLM_model_name"]
        else:
            model_name = "gemini-1.5-pro"

        self.google_api_key = None
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        import vertexai.preview.generative_models as generative_models
        if "GOOGLE_APPLICATION_CREDENTIALS" in config or os.environ["GOOGLE_APPLICATION_CREDENTIALS"]:
            """
            If Google api_key is provided through config
            or set as an environment variable, assign it.
            """
            vertexai.init(project=config["gcp_project"], location=config["gcp_location"])

            # genai.configure(api_key=config["api_key"])
            # self.chat_model = genai.GenerativeModel(model_name)
            self.chat_model = GenerativeModel(model_name)
        else:
            # Authenticate using VertexAI
            
            self.chat_model = GenerativeModel(model_name)

    def system_message(self, message: str) -> any:
        return message

    def user_message(self, message: str) -> any:
        return message

    def assistant_message(self, message: str) -> any:
        return message

    def submit_prompt(self, prompt, **kwargs) -> str:
        response = self.chat_model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
            },
        )
        return response.text
