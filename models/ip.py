import pandas as pd
from config import file_paths
from pymer4.models import Lmer
from utils import load_data, rename_columns
from visualizations import plot_mean_with_ci, plot_coefficients_1, plot_tukey_hsd   

"""
Investigate the length of Intonational Phrases (IPs) by each speaker group (bilingual vs. monolingual) 
with factors such as formality.
"""

# Load and preprocess the data
file_path = file_paths["ip_model"]
data = load_data(file_path)
data = rename_columns(data)

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].map(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].map(lambda x: 1 if x == 'formal' else -1)

# Define the GLMM formula
formula = ' IP_length  ~ bilingual_contrast * formality_contrast + (1|speaker_id)'

# Fit the GLMM using pymer4's Lmer
glmm_model = Lmer(formula, data=data)    
glmm_model.fit(REML=False)  # Using MLE instead of REML

# Print the model summary
print("GLMM Summary:")
print(glmm_model.summary())

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

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