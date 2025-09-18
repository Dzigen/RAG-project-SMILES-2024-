import re
from abc import ABC, abstractmethod
from mistralai import Mistral


class BaseLMJudge(ABC):

    def __init__(self, client, args: dict[str, dict]):
        self.args = args
        self.client = client(**args["client_args"])

    @abstractmethod
    def __call__(self, gen_args):
        pass


class MistralJudge(BaseLMJudge):

    def __init__(self, client=Mistral, args: dict[str, dict] = None):
        super().__init__(client=client, args=args)

    def __call__(self, gen_args: dict):
        return self._call_mistral(**gen_args)
        # return self._parse_score(self._call_mistral(**gen_args))
    
    def _call_mistral(
        self,
        model_name: str = "mistral-large-latest",
        system_prompt: str = None,
        user_prompt: str = None,
        assistant_prompt: str = None,
        gen_config: dict = None
    ) -> str:
    
        messages = []
        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        if assistant_prompt is not None:
            messages.append({
                "role": "assistant",
                "content": assistant_prompt
            })
            
        if user_prompt is not None:
            messages.append({
                "role": "user",
                "content": user_prompt
            });
    
        if gen_config is None:
            gen_config = {}
            
        chat_response = self.client.chat.complete(
            model=model_name,
            messages=messages,
            **gen_config
        )
    
        return chat_response.choices[0].message.content

    def _parse_score(self, raw_output: str) -> int:
        pattern = r'.*Оценка модели: (\d).*'
        score = re.match(pattern, repr(raw_output)).group(1)
        return int(score)