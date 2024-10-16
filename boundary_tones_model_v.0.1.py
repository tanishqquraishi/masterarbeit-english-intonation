import pandas as pd
from pymer4.models import Lmer
from utils import load_data, rename_columns, create_binary_column
from boundary_tones_VIS import plot_coefficients, plot_likelihood_by_group, plot_interaction_bilingual_gender  # Import visualizations

# Load data
file_path = r"C:\\Users\\Tanishq\\Documents\\stuttgart\\Study\\thesis\\data\\model data\\boundary_tones_17.07.2024.xlsx"
data = load_data(file_path)
data = rename_columns(data)

# Define lists for high and low BT
high_boundary_tones = ['H-H%', 'L-H%', 'H-', 'H-^H%', 'L-^H%', '!H-H%', '^H-H%']
low_boundary_tones = ['L-L%', 'H-L%', 'L-', '!H-L%', '^H-L%']

# Create a binary column for high and low BT
def boundary_tone_condition(row):
    if row['boundary_tone'] in high_boundary_tones:
        return 1
    elif row['boundary_tone'] in low_boundary_tones:
        return 0
    return None

data = create_binary_column(data, 'boundary_tone_binary', boundary_tone_condition)

# Drop rows with None values in the boundary_tone_binary column
data = data.dropna(subset=['boundary_tone_binary'])

# Contrast-code the independent variables
data['bilingual_contrast'] = data['bilingual'].apply(lambda x: 1 if x == 'yes' else -1)
data['formality_contrast'] = data['formality'].apply(lambda x: 1 if x == 'formal' else -1)
data['gender_contrast'] = data['gender'].apply(lambda x: 1 if x == 'female' else -1)

# Interaction terms
data['interaction_formality_gender'] = data['formality_contrast'] * data['gender_contrast']
data['interaction_bilingual_gender'] = data['bilingual_contrast'] * data['gender_contrast']
data['interaction_bilingual_formality'] = data['bilingual_contrast'] * data['formality_contrast']

# Define the generalized linear mixed model using pymer4
formula = 'boundary_tone_binary ~ bilingual_contrast + formality_contrast + gender_contrast + interaction_bilingual_formality + interaction_bilingual_gender + interaction_formality_gender + (1|speaker_id)'

# Fit the GLMM model using a binomial family
glmm_model = Lmer(formula, data=data, family='binomial')
glmm_model.fit()

# Print the summary of the GLMM model
print(glmm_model.summary())

# Store the fitted values in the DataFrame
data['fittedvalues'] = glmm_model.predict(data)

# Visualization of fixed effects coefficients with error bars
model_data = {
    'Variable': glmm_model.coefs['Estimate'].index,
    'Coef.': glmm_model.coefs['Estimate'].values,
    'Std.Err.': glmm_model.coefs['Std. Error'].values,
}

model_df = pd.DataFrame(model_data)

################## Visualizations ##################
# Call vis file func
plot_coefficients(model_df)  # Plot the fixed effects coefficients
plot_likelihood_by_group(data, 'formality', 'Likelihood of High Boundary Tone by Formality', 'Likelihood of High Boundary Tone (%)')
plot_likelihood_by_group(data, 'gender', 'Likelihood of High Boundary Tone by Gender', 'Likelihood of High Boundary Tone (%)')
plot_interaction_bilingual_gender(data)