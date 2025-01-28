import pandas as pd
from config import file_paths
from utils import load_data, rename_columns, create_binary_column
from visualizations import plot_coefficients
from pymer4.models import Lmer

"""
Investigate the length of IP by each speaker group (bilingual vs. monolingual speakers) with factors 
such as formality.
"""

# Load and preprocess the data
file_path = file_paths["ip_model"]
data = load_data(file_path)
data = rename_columns(data)

data = create_binary_column(
    data=data,
    column_name='ip_boundary',
    condition=lambda row: pd.notna(row['word_bt'])
)

# Assign unique IP IDs based on boundary tones
ip_id = 1
ip_ids = []
for boundary in data['ip_boundary']:
    ip_ids.append(ip_id)
    if boundary:
        ip_id += 1
data['ip_id'] = ip_ids

# Calculate IP lengths (number of words per IP)
ip_lengths = data.groupby('ip_id').size().reset_index(name='ip_length')

# Merge IP lengths back into the original data
#data = data.merge(ip_lengths, on='ip_id', how='left')

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].map(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].map(lambda x: 1 if x == 'formal' else -1)

# Define the formula for the GLMM
formula = 'ip_length ~ bilingual_contrast * formality_contrast + (1|speaker_id)'

# Fit the GLMM using pymer4's Lmer
model = Lmer(formula, data=data)    
model.fit(REML=False) #using MLE instead of REML

# Print the model summary
print("GLMM Summary:")
print(model.summary())

data['fittedvalues'] = model.predict(data, skip_data_checks=True, verify_predictions=False)

######## Extract z and p scores for reporting #############
#z_and_p_values = model.coefs[['Z-stat', 'P-val']]

# Rename the index for better readability
#z_and_p_values.index.name = 'Effect'

# Customize p-value formatting
#z_and_p_values['P-val'] = z_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
#print("Z-scores and P-values for Fixed Effects:")
#print(z_and_p_values)

################## Visualizations ##################

# Prepare the model coefficients DataFrame
model_data = {
    'Variable': model.coefs['Estimate'].index,
    'Coef.': model.coefs['Estimate'].values,
    'Std.Err.': model.coefs['SE'].values,
}

# Visualize the model coefficients
model_df = pd.DataFrame(model_data)
plot_coefficients(model_df, title="GLMM Coefficients for IP Length")


# Visualize the average IP lengths by speaker group
#plot_likelihood_by_group(
#    data=data, 
#    group_col='bilingual', 
#    title='Average IP Length by Speaker Group', 
#    ylabel='Average IP Length'
#)