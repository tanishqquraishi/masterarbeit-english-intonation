import pandas as pd
from config import file_paths

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # comment out to see warning regarding pymer4 version deprecation

from pymer4.models import Lmer
from utils import load_data, rename_columns, convert_mixed_columns_to_string
from visualizations import plot_mean_with_ci, plot_coefficients_1, plot_tukey_hsd , display_model_fit, display_fixed_effects  

"""
Investigate the length of Intonational Phrases (IPs) by each speaker group (bilingual vs. monolingual) 
with factors such as formality.
"""

# Load and preprocess the data
file_path = file_paths["ip_model"]
data = load_data(file_path)
data = rename_columns(data)
# Convert mixed-type columns to string
data = convert_mixed_columns_to_string(data)

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].map(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].map(lambda x: 1 if x == 'formal' else -1)

# Define the GLMM formula
formula = ' IP_length  ~ bilingual_contrast * formality_contrast + (1|speaker_id)'

# Fit the GLMM using pymer4's Lmer
glmm_model = Lmer(formula, data=data)    
glmm_model.fit(REML=False)  # Using MLE instead of REML

# Print the model summary
# print("GLMM Summary:")
print(glmm_model.summary())

# Display Model Fit Statistics
display_model_fit(glmm_model)

# Display Fixed Effects Coefficients
display_fixed_effects(glmm_model)

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

######## Extract t and p scores for reporting #############

# Extract T-stat and P-val from the model coefficients
t_and_p_values = glmm_model.coefs[['T-stat', 'P-val']]

# Rename the index for better readability
t_and_p_values.index.name = 'Effect'

# Customize p-value formatting
t_and_p_values.loc[:, 'P-val'] = t_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
print("T-scores and P-values for Fixed Effects Extracted for Reporting:")
print(t_and_p_values)

################## Visualizations ##################

# Prepare the model coefficients DataFrame
model_data = {
    'Variable': glmm_model.coefs['Estimate'].index,
    'Coef.': glmm_model.coefs['Estimate'].values,
    '2.5_ci': glmm_model.coefs['2.5_ci'].values,
    '97.5_ci': glmm_model.coefs['97.5_ci'].values,
    'Std.Err.': glmm_model.coefs['SE'].values,
}

# Visualize the model coefficients with confidence intervals
model_df = pd.DataFrame(model_data)
plot_coefficients_1(model_df, title="GLMM Coefficients for IP Length")

# Visualize the mean IP length by group with confidence intervals
plot_mean_with_ci(data, 'bilingual', 'Mean IP Length by Speaker Group', 'Mean IP Length')
plot_mean_with_ci(data, 'formality', 'Mean IP Length by Formality', 'Mean IP Length')
#plot_tukey_hsd(data, 'bilingual_formality', 'IP_length')