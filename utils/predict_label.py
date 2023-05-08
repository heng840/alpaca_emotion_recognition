import pdb
from datasets import load_dataset
import pandas
import re


def predict_label(generator):
    result = ''.join(result for result in generator)
    # print('result:', result)
    label = None
    # Use regex to extract the label from the result string
    # pdb.set_trace()
    label_match = re.search(r"'label'\s*:\s*(\d+)", result)
    if label_match:
        label = int(label_match.group(1))
    else:
        # Try to directly parse the entire result string as an integer
        try:
            label = int(result.strip())
        except ValueError:
            # Extract the last character of the result string as the label
            last_char = result.strip()[-1]
            if last_char.isdigit():
                label = int(last_char)
            else:
                # If the previous method fails, try to find the label using the new method
                emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]
                for idx, emotion in enumerate(emotions):
                    if emotion in result:
                        label = idx

    return label


def calculate_accuracy(predicted_labels, true_labels):
    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = correct_predictions / len(true_labels)
    return accuracy


if __name__ == "__main__":
    dataset = load_dataset("dair-ai/emotion")
    # Data Fields
    # text: a string feature.
    # label: a classification label, with possible values including:
    # sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).
    train_data = dataset["train"]
    validation_data = dataset["validation"]
    validation_df = validation_data.to_pandas()

    # Example dataset
    dataset = [
        {"input": "Text example 1", "label": 0},
        {"input": "Text example 2", "label": 1},
        # ...
    ]

    # Predict labels for the dataset
    predicted_labels = [predict_label(example["input"]) for example in dataset]

    # Remove examples where the label prediction failed (None values)
    filtered_dataset = [example for example, pred_label in zip(dataset, predicted_labels) if pred_label is not None]

    # Extract true labels for the filtered dataset
    true_labels = [example["label"] for example in filtered_dataset]

    # Calculate the accuracy rate
    accuracy = calculate_accuracy(predicted_labels, true_labels)
    print("Accuracy:", accuracy)
