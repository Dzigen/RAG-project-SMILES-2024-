import re


def make_prompt(user_prompt, 
                question, 
                gt, 
                gen_ans, 
                ending="Model score:", 
                prompt_sep="\n\n"):
    prompt = (user_prompt + 
              prompt_sep + 
              question + "\n" + 
              gt + "\n" + 
              gen_ans + "\n" +
              ending)
    return prompt


def parse_gen(gen: str, endings: list):
    score = '-1'
    for ending in endings:
        if score == '-1':
            ending_idx = gen.find(ending)
            if ending_idx > -1:
                start_idx = ending_idx + len(ending)
                if '0' in gen[start_idx:start_idx + 10]:
                    score = '0'
                elif '1' in gen[start_idx:start_idx + 10]:
                    score = '1'
            if score == '-1':
                if '0' in gen[:5] + gen[-5:]:
                    score = '0'
                elif '1' in gen[:5] + gen[-5:]:
                    score = '1'
        else:
            break
    return int(score)