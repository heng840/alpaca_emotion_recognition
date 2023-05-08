import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create a dictionary with the results
results = {
    'Model': ['7B', '7B', '7B', '7B', '7B', '7B', '13B', '13B', '13B', '13B', '13B', '13B'],
    'Fine-tuned on': ['Direct', '100', '200', '300', '400', '500', 'Direct', '100', '200', '300', '400', '500'],
    'Accuracy': [0.25499729875742844, 0.3534, 0.3755, 0.3904, 0.3930, 0.3931,
                 0.45, 0.66112, 0.68542, 0.68785, 0.68543, 0.68784]
}

# Create a DataFrame with the results
df = pd.DataFrame(results)

# Save the table to a CSV file
df.to_csv('results_table.csv', index=False)

# Create a bar plot comparing the results
bar_width = 0.35
opacity = 0.8
fine_tuned_data = df['Fine-tuned on'].unique()
num_bars = len(fine_tuned_data)
index = np.arange(num_bars)

plt.figure(figsize=(10, 5))
for i, model in enumerate(df['Model'].unique()):
    plt.bar(index + i * bar_width, df[df['Model'] == model]['Accuracy'], bar_width, label=model, alpha=opacity)

plt.xlabel('Fine-tuned on')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.xticks(index + bar_width / 2, fine_tuned_data)
plt.legend()

# Save the graph as an image file
plt.savefig('model_comparison.png', dpi=300)

# Show the graph
plt.show()
