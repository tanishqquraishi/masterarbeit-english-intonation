# -*- coding: utf-8 -*-
"""boundary_tones_model v.0.1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZBJNeQpxdwb9yxbQ02JL_HUODDcecOEn

# Data and Libraries 📚
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

file_path = "/content/boundary_tones_17.07.2024.xlsx"

data = pd.read_excel(file_path)

pd.DataFrame(data)

# Rename the columns to more straightforward names
data = data.rename(columns={
    '1_anno_default_ns:bt': 'boundary_tone',
    '1_meta_speaker-bilingual': 'bilingual',
    '1_meta_setting': 'formality',
    '1_meta_speaker-gender': 'gender',
    '1_meta_speaker-id': 'speaker_id'
})

data.head(2)

data['boundary_tone'].value_counts()

# Define lists for high and low boundary tones
high_boundary_tones = [
    'H-H%', 'L-H%', 'H-', 'H-^H%', 'L-^H%', '!H-H%', '^H-H%'
]
low_boundary_tones = [
    'L-L%', 'H-L%', 'L-', '!H-L%', '^H-L%'
]

# Create a binary column for high and low boundary tones
data['boundary_tone_binary'] = data['boundary_tone'].apply(
    lambda x: 1 if x in high_boundary_tones else (0 if x in low_boundary_tones else None)
)

# Drop rows with None values in the boundary_tone_binary column
data = data.dropna(subset=['boundary_tone_binary'])

print(data[['boundary_tone', 'boundary_tone_binary']].head())

"""# Model 🔢"""

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Interaction terms
data['interaction_formality_gender'] = data['formality_contrast'] * data['gender_contrast']
data['interaction_bilingual_gender'] = data['bilingual_contrast'] * data['gender_contrast']
data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the fixed effects formula
formula = 'boundary_tone_binary ~ bilingual_contrast * formality_contrast * gender_contrast'

# Fit the GLM model
glm_model = smf.glm(formula, data=data, family=sm.families.Binomial()).fit()

# Print the summary of the GLM model
print(glm_model.summary())

"""Account for the speaker-level random effects using a custom approach. The AUC (Area Under the Receiver Operating Characteristic Curve) score is a metric used to evaluate the performance of a binary classification model."""

# Extract fixed effect predictions
data['fixed_effects'] = glm_model.fittedvalues

# Group by speaker_id to calculate random effects
speaker_means = data.groupby('speaker_id')['boundary_tone_binary'].mean()
overall_mean = data['boundary_tone_binary'].mean()

# Calculate random effects as deviations from the overall mean
data['random_effects'] = data['speaker_id'].map(speaker_means - overall_mean)

# Adjust predictions by adding random effects
data['adjusted_predictions'] = data['fixed_effects'] + data['random_effects']

# Ensure adjusted predictions are in the valid range for probabilities
data['adjusted_predictions'] = data['adjusted_predictions'].apply(lambda x: min(max(x, 0), 1))

# Evaluate the model


from sklearn.metrics import roc_auc_score

# AUC scores range from 0 to 1. A score closer to 1
# indicates that the model has better discriminatory ability,
# meaning it can distinguish between the two choices of boundary tones more effectively.

auc_score = roc_auc_score(data['boundary_tone_binary'], data['adjusted_predictions'])
print(f"AUC Score: {auc_score}")

"""#Visualization 📊



"""

# Data from the model
model_data = {
    'Variable': [
        'Intercept', 'bilingual_contrast', 'formality_contrast', 'bilingual_contrast:formality_contrast',
        'gender_contrast', 'bilingual_contrast:gender_contrast', 'formality_contrast:gender_contrast',
        'bilingual_contrast:formality_contrast:gender_contrast'
    ],
    'Coef.': [-1.2225, 0.0261, 0.0683, 0.0247, 0.2278, 0.0454, 0.0076, -0.0836],
    'Std.Err.': [0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032, 0.032],
    'P>|z|': [0.000, 0.409, 0.030, 0.433, 0.000, 0.150, 0.810, 0.008],
    '95% CI Lower': [-1.284, -0.036, 0.006, -0.037, 0.166, -0.016, -0.054, -0.145],
    '95% CI Upper': [-1.161, 0.088, 0.130, 0.087, 0.290, 0.107, 0.069, -0.022]
}

# Create DataFrame
model_df = pd.DataFrame(model_data)

# Visualize the coefficients
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting the coefficients with error bars
ax.errorbar(model_df['Variable'], model_df['Coef.'], yerr=model_df['Std.Err.'], fmt='o', color='blue', ecolor='black', capsize=5)

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='grey', linestyle='--')

# Add labels and title
ax.set_xlabel('Variable')
ax.set_ylabel('Coefficient')
ax.set_title('GLM Coefficients with 95% Conf Interval for BT Binary')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Group data by formality and calculate mean of adjusted predictions (likelihood of high boundary tone)
# Assuming 'data' is your DataFrame and 'adjusted_predictions' is a column in it
formality_likelihood = data.groupby('formality')['adjusted_predictions'].mean()

# Create bar plot
plt.figure(figsize=(8, 6))
formality_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Formality')
plt.ylabel('Likelihood of High Boundary Tone (%)')
plt.title('Likelihood of High Boundary Tone by Formality')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal

# Function to format y-axis as percentage
def to_percent(y, position):
    return f'{y * 100:.0f}%'

# Apply the percentage formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.tight_layout()
plt.show()

gender_likelihood = data.groupby('gender')['adjusted_predictions'].mean()

# Create bar plot
plt.figure(figsize=(8, 6))
bars = gender_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])

# Set labels and title
plt.xlabel('Gender')
plt.ylabel('Likelihood of High Boundary Tone (%)')
plt.title('Likelihood of High Boundary Tone by Gender')
plt.xticks(rotation=0)  # Keep x-axis labels horizontal

# Apply the percentage formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

plt.tight_layout()
plt.show()

!apt-get install git

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/tanishqquraishi/masterarbeit-english-intonation.git
# %cd masterarbeit-english-intonation

# Commented out IPython magic to ensure Python compatibility.
import os

# Store the token in a variable
GITHUB_TOKEN = "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"

# Clone the repository using the token
!git clone https://{GITHUB_TOKEN}@github.com/tanishqquraishi/masterarbeit-english-intonation.git
# %cd masterarbeit-english-intonation

