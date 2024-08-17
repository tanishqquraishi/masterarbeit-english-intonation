import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter 
from utils import load_data, rename_columns, create_binary_column

# Load the data
file_path = r"C:\Users\Tanishq\Documents\stuttgart\Study\thesis\data\model data\ip_data_17.07.2024.xlsx"
data = load_data(file_path)

# Step 1: Preprocessing
# Add a column that marks 1 for every word_bt present and 0 for NaN values
data = create_binary_column(data, 'bt_flag', lambda row: 1 if pd.notnull(row['2_anno_default_ns:word_bt']) else 0)

# Create an identifier for each IP such that each IP ends at the row where bt_flag is 1
data['ip_id'] = data['bt_flag'].cumsum()

# Calculate the length of each IP (number of words in the IP)
ip_lengths = data.groupby(['ip_id', '1_meta_speaker-id', '1_meta_speaker-bilingual', '1_meta_setting']).size().reset_index(name='ip_length')

# Rename columns for clarity
ip_lengths = rename_columns(ip_lengths)

# Contrast-code the independent variables
ip_lengths['bilingual_contrast'] = ip_lengths['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
ip_lengths['formality_contrast'] = ip_lengths['formality'].apply(lambda x: 1 if x == 'formal' else -1)

# Step 2: Define the linear mixed model formula with interaction terms
formula = 'ip_length ~ bilingual_contrast * formality_contrast'

# Fit the Linear Mixed Model with random intercept for speaker_id
mixedlm_model = smf.mixedlm(formula, ip_lengths, groups=ip_lengths["speaker_id"]).fit()

# Print the summary of the Linear Mixed Model
print(mixedlm_model.summary())

# Step 3: Visualize the model results
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
ax.set_title('Linear Mixed Model Coefficients with 95% Conf Interval for IP Length')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()