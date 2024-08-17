import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter 
from utils import load_data, rename_columns, create_binary_column

# Load the data
file_path = r"C:\Users\Tanishq\Documents\stuttgart\Study\thesis\data\model data\pitch_accents_17.07.2024.xlsx"  
data = load_data(file_path)

# Step 1: Create a binary column for pitch accents
monotonal_pitch_accents = ["H*", "L*"]
bitonal_pitch_accents = ["L+H*", "L*+H", "H+L*", "H*+!H", "H*+L", "H+!H*", "H*+L"]

data = create_binary_column(
    data, 
    'pa_type_binary', 
    lambda row: 1 if row['2_anno_default_ns:word_pa'] in monotonal_pitch_accents else 
                (0 if row['2_anno_default_ns:word_pa'] in bitonal_pitch_accents else None)
)

# Drop rows with None values in the pa_type_binary column
data = data.dropna(subset=['pa_type_binary'])

# Step 2: Rename the columns for clarity
data = rename_columns(data)

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Step 3: Define the mixed effects model formula with interaction terms
formula = 'pa_type_binary ~ bilingual_contrast * formality_contrast + gender_contrast'

# Fit the Mixed Linear Model with random intercept for speaker_id
mixedlm_model = smf.mixedlm(formula, data, groups=data["speaker_id"]).fit()

# Print the summary of the MixedLM model
print(mixedlm_model.summary())

# Step 4: Plot the model results
# Visualization of fixed effects coefficients with error bars
model_data = {
    'Variable': mixedlm_model.params.index,
    'Coef.': mixedlm_model.params.values,
    'Std.Err.': mixedlm_model.bse.values,
}

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
ax.set_title('MixedLM Coefficients with 95% Conf Interval for Monotonal vs Bitonal PA')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 5: Plot the likelihood of a monotonal PA based on speaker group
# Calculate the mean likelihood of monotonal PA by speaker group
data['predicted_pa'] = mixedlm_model.fittedvalues
group_likelihood = data.groupby('bilingual')['predicted_pa'].mean()

# Create bar plot for speaker group likelihood
plt.figure(figsize=(8, 6))
group_likelihood.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Speaker Group')
plt.ylabel('Likelihood of Monotonal PA (%)')
plt.title('Likelihood of Monotonal PA by Speaker Group')
plt.xticks(rotation=0)

# Apply the percentage formatter to the y-axis
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.tight_layout()
plt.show()