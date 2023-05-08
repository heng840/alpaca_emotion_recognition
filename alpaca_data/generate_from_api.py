import openai
import json

openai.api_key = 'your_openai_api_key'

def generate_text_and_label(prompt):
    response = openai.Completion.create(
        engine='gpt-3.5-turbo',
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.8,
    )
    return response.choices[0].text.strip()

def main():
    instruction = "The input will be a text.You need to sentiment classify it." \
                  "There are five types of preset labels:sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)." \
                  "Some examples look as follows." \
                  "'text': 'im feeling quite sad and sorry for myself but ill snap out of it soon','label': 0," \
                  "'text': i am feeling pretty good today, 'label': 1,"

    dataset = []

    for i in range(10):
        prompt = f"{instruction} Generate a text and its corresponding label:"
        result = generate_text_and_label(prompt)
        text, label = result.split(', ')
        label = int(label.split(':')[-1])
        dataset.append({'text': text, 'label': label})

    with open('emotion_data.json', 'w') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()
