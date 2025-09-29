import json
from utils import load_jsonl, save_jsonl, split_prefix
import argparse
from datasets import load_dataset


def main():
    # conver to dict_keys(['query_id', 'verify', 'prompt', 'final_answer', 'answer'])
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./OpenR1-50-0-4.jsonl')
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        for i, line in enumerate(f):
            d = None
            try:
                d = json.loads(line)
            except:
                print(f"Error in line {i}")
                continue
            text = d['generation']
            if text[0] == "\"":
                text = text[1:]
                text = text[:-1]
            solution = text.split('</think>')[-1]
            prefix, suffix = split_prefix(solution, 0.25)
            if len(prefix) < 10:
                prompt = d['problem'] + '\n\n'
            else:
                prompt = d['problem'] + '\n\n' + '## Hint.' + prefix
            answer = text
            final_answer = d['answer']
            # final_answer = d['expected_answer']
            if final_answer not in solution:
                continue
            new_d = {}
            new_d['query_id'] = f"{i}"
            new_d['verify'] = True
            new_d['prompt'] = prompt
            new_d['task'] = 'math'
            new_d['solutions'] = ['\\boxed{'+final_answer+'}']
            with open('OpenR1-50-0-4-prefix.jsonl', "a") as A:
                A.write(json.dumps((new_d), ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
