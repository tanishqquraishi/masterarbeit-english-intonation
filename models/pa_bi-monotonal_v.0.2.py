import pandas as pd
from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column
from visualizations_binomial_models import plot_coefficients, plot_likelihood_by_group

# Load and preprocess the data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\pitch_accents_17.07.2024.xlsx"
data = load_data(file_path)
data = rename_columns(data)

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
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Create interaction terms
data['interaction_formality_gender'] = data['formality_contrast'] * data['gender_contrast']
data['interaction_bilingual_gender'] = data['bilingual_contrast'] * data['gender_contrast']
data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the GLMM formula
formula = 'pa_type_binary ~ bilingual_contrast + formality_contrast + gender_contrast + interaction_bilingual_formality + interaction_bilingual_gender + interaction_formality_gender + (1|speaker_id)'

# Fit the GLMM model using a binomial family
glmm_model = Lmer(formula, data=data, family='binomial')
glmm_model.fit()

# Print the model summary
print(glmm_model.summary())

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data, skip_data_checks=True, verify_predictions=False)

# Extract only z and p scores 

# Extract Z-stat and P-val from the model coefficients
z_and_p_values = glmm_model.coefs[['Z-stat', 'P-val']]

# Rename the index for better readability
z_and_p_values.index.name = 'Effect'

# Customize p-value formatting
z_and_p_values['P-val'] = z_and_p_values['P-val'].apply(lambda x: "<0.001" if x < 0.001 else round(x, 3))
print("Z-scores and P-values for Fixed Effects: ")
print(z_and_p_values)

################## Visualizations ##################

# Visualize coefficients
model_data = pd.DataFrame({
    'Variable': glmm_model.coefs.index,
    'Coef.': glmm_model.coefs['Estimate'],
    'Std.Err.': glmm_model.coefs['SE']
})
plot_coefficients(model_data, title='GLMM Coefficients for Monotonal vs Bitonal Pitch Accents')

# Visualizations of likelihood by group
plot_likelihood_by_group(
    data, 'bilingual', 
    title='Likelihood of Bitonal Pitch Accent by Speaker Group', 
    ylabel='Likelihood of Bitonal Pitch Accent (%)'
)

plot_likelihood_by_group(
    data, 'formality', 
    title='Likelihood of Bitonal Pitch Accent by Formality', 
    ylabel='Likelihood of Bitonal Pitch Accent (%)'
)

plot_likelihood_by_group(
    data, 'gender', 
    title='Likelihood of Bitonal Pitch Accent by Gender', 
    ylabel='Likelihood of Bitonal Pitch Accent (%)'
)
