import pandas as pd
from config import file_paths

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # comment out to see warning regarding pymer4 version deprecation

from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column, convert_mixed_columns_to_string
from visualizations import plot_coefficients_1, plot_mean_with_ci, display_model_fit, display_fixed_effects

"""
Investigate the likelihood of a monotonal or bitonal PA by each speaker group (bilingual vs. monolingual speakers) 
with factors such as formality.
"""

# Load and preprocess the data
file_path = file_paths["pa_model"]
data = load_data(file_path)
data = rename_columns(data)
# Convert mixed-type columns to string
data = convert_mixed_columns_to_string(data)

# Define monotonal and bitonal pitch accents
monotonal_pitch_accents = ["H*", "L*"]
bitonal_pitch_accents = ["L+H*", "L*+H", "H+L*", "H*+!H", "H*+L", "H+!H*", "H*+L"]

# Create a binary column for pitch accents (1 = monotonal, 0 = bitonal)
data = create_binary_column(
    data, 
    'pa_type_binary', 
    lambda row: 1 if row['word_pa'] in monotonal_pitch_accents else 
                (0 if row['word_pa'] in bitonal_pitch_accents else None)
)

# Drop rows with None values in the pa_type_binary column
data = data.dropna(subset=['pa_type_binary'])

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)


data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the GLMM formula
formula = 'pa_type_binary ~ bilingual_contrast + formality_contrast + interaction_bilingual_formality + (1|speaker_id)'

# Fit the GLMM model using a binomial family
glmm_model = Lmer(formula, data=data, family='binomial')
glmm_model.fit()

# Print the model summary
# print(glmm_model.summary())

# Display Model Fit Statistics
display_model_fit(glmm_model)

# Display Fixed Effects Coefficients
display_fixed_effects(glmm_model)

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

######## Extract z and p scores for reporting #############

# Extract Z-stat and P-val from the model coefficients
z_and_p_values = glmm_model.coefs[['Z-stat', 'P-val']]

# Rename the index for better readability
z_and_p_values.index.name = 'Effect'

# Customize p-value formatting
z_and_p_values.loc[:, 'P-val'] = z_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
print("Z-scores and P-values for Fixed Effects Extracted for Reporting: ")
print(z_and_p_values)


################ Visualizations ####################3
model_data = pd.DataFrame({
    'Variable': glmm_model.coefs.index,
    'Coef.': glmm_model.coefs['Estimate'],
    '2.5_ci': glmm_model.coefs['2.5_ci'],
    '97.5_ci': glmm_model.coefs['97.5_ci'],
    'Std.Err.': glmm_model.coefs['SE']
})
plot_coefficients_1(model_data, title='GLMM Coefficients for Monotonal vs Bitonal Pitch Accents')

# Visualizations of likelihood by group
plot_mean_with_ci(
    data, 'bilingual', 
    title='Likelihood of Bitonal Pitch Accent by Speaker Group', 
    ylabel='Likelihood of Bitonal Pitch Accent'
)

plot_mean_with_ci(
    data, 'formality', 
    title='Likelihood of Bitonal Pitch Accent by Formality', 
    ylabel='Likelihood of Bitonal Pitch Accent'
)