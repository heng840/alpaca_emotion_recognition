import json
import re


def process_line(line):
    text_match = re.search("'text': (.*?), 'label': (\d)", line)

    if text_match:
        input_text = text_match.group(1).strip()
        output_label = text_match.group(2).strip()

        return {
            "instruction": "The input will be a text. You need to sentiment classify it. There are five types of preset labels: sadness (0), joy (1), love (2), anger (3), fear (4 ), surprise (5). Some examples looks as follows: 'text': 'im feeling quite sad and sorry for myself but ill snap out of it soon','label': 0, 'text': i am feeling pretty good today, 'label': 1,",
            "input": input_text,
            "output": output_label
        }
    else:
        return None


with open('GPT_emotion_data.txt', 'r') as file:
    lines = file.readlines()

json_data = []

for line in lines:
    processed_line = process_line(line)
    if processed_line:
        json_data.append(processed_line)

with open('GPT_emotion_data.json', 'w') as outfile:
    json.dump(json_data, outfile, indent=4)
