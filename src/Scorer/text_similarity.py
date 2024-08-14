from .utils import SimilarityScorerConfig
from src.Reader import LLM_Model

from Levenshtein import distance
from typing import List

class LLM_SimilarityScorer:
    def __init__(self, config: SimilarityScorerConfig, reader: LLM_Model):
        self.config = config
        self.reader = reader
        self.system_prompt = self.config.system_prompt.format(
            sim_label=self.config.sim_label,dissim_label=self.config.dissim_label)

    def predict(self, txts1: List[str], txts2: List[str]):
        user_prompts = [self.config.user_prompt_template.format(txt1=t1, txt2=t2) for t1, t2 in zip(txts1, txts2)]
        output = self.reader.generate(user_prompts, system_prompt=self.system_prompt, 
                                      gen_params=self.config.gen, batch_size=self.config.batch_size)

        predicted_scores = []
        for raw_score in output:
            print("sim raw len: ", len(raw_score))
            
            score = raw_score[0].lower()
            sim_dist = distance(self.config.sim_label, score)
            dissim_dist = distance(self.config.dissim_label, score)
            predicted_scores.append(1 if sim_dist < dissim_dist else 0)

        return predicted_scores