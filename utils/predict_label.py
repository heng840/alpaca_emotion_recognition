from datasets import load_dataset
import pandas
import re


# 定义一个函数，该函数调用 evaluate 并从结果中提取标签
def predict_label(input_text, generator):
    result = ''.join(result for result in generator)

    # 从结果字符串中提取标签
    label_str = result.split("label:")[-1].strip()
    label = int(label_str)

    return label


def cal_acc(dataset):
    correct_predictions = 0
    total_predictions = 0

    for data_point in dataset:
        input_text = data_point["input"]
        true_label = data_point["true_label"]

        predicted_label = predict_label(input_text)

        if predicted_label == true_label:
            correct_predictions += 1

        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.2%}")

dataset = load_dataset("dair-ai/emotion")
# Data Fields
# text: a string feature.
# label: a classification label, with possible values including:
# sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
train_data = dataset["train"]
validation_data = dataset["validation"]
validation_df = validation_data.to_pandas()