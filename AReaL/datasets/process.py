import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./OpenR1-50-0-4-prefix.jsonl')
    parser.add_argument("--output", type=str, default='data/train.jsonl')
    args = parser.parse_args()


    with open(args.input, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            new_data = {}
            new_data['question'] = data['prompt']
            new_data['answer'] = data['solutions']
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_data, ensure_ascii=False) + '\n')
