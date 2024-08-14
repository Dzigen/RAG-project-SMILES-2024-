from .utils import UncertaintyScorerConfig
from src.Reader import LLM_Model

from Levenshtein import distance
from typing import List

class LLM_UncertaintyScorer:
    def __init__(self, config: UncertaintyScorerConfig, reader: LLM_Model):
        self.config = config
        self.reader = reader
        self.system_prompt = self.config.system_prompt.format(
            cert_label=self.config.cert_label,uncert_label=self.config.uncert_label)

    def predict(self, query: str, contexts: List[str]):
        user_prompts = [self.config.user_prompt_template.format(q=query, c=cntx) for cntx in contexts]
        output = self.reader.generate(user_prompts, system_prompt=self.system_prompt, 
                                      gen_params=self.config.gen, batch_size=self.config.batch_size)

        predicted_scores = []
        for raw_score in output:
            print("uncert raw len: ", len(raw_score))
            
            score = raw_score[0].lower()
            cert_dist = distance(self.config.cert_label, score)
            unsert_dist = distance(self.config.uncert_label, score)
            predicted_scores.append(1 if cert_dist < unsert_dist else 0)

        return predicted_scores