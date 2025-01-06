from .gptcall import request as gptcall
from .tgtcall import request as tgtcall

class LLM():
    def __init__(self, openai_api_key, together_api_key):
        self.openai_api_key = openai_api_key
        self.together_api_key = together_api_key

    def __call__(self, prompt, model='gpt-4o-mini', system_message=None, **kwargs):
        if model in ['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini']:
            return gptcall(prompt, model=model, system_message=system_message, api_key=self.openai_api_key, **kwargs)
        elif '/' in model:
            return tgtcall(prompt, model=model, system_message=system_message, api_key=self.together_api_key, **kwargs)
        else:
            raise NotImplementedError("Model not supported")
