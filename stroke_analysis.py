import pandas  # Imports the csv data

from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn


df: DataFrame = pandas.read_csv("healthcare-dataset-stroke-data.csv")
df_array = df.to_numpy()

# Discarding not numerical values in bmi
discarding = pandas.isna(df["bmi"])
# Extracting the names of the head in the dataframe:


print(100 * '-')
# Análisis lógico
Stroked_patients      = df_array[:, -1] == 1
Diseased_patients     = df_array[:, 4] == 1
NStroked_patients     = sum(Stroked_patients)
NNon_stroked_patients = sum(df_array[:, -1] == 0)
Males                 = (df_array[:, 1] == "Male") | (df_array[:, 1] == "Other")
Females               = df_array[:, 1] == "Female"
bmi_incomplete        = discarding.to_numpy()

print(sum(bmi_incomplete))
P_stroked_pacients = NStroked_patients / (NNon_stroked_patients + NStroked_patients)

Smokers = (df_array[:, 10] == 'smokes') | (df_array[:, 10] == 'formerly smoked')
N_Smokers = sum(Smokers)
Stroked_Smokers = (Smokers) & (Stroked_patients)
N_Stroked_smokers = sum(Stroked_Smokers)

P_stroked_smokers = N_Stroked_smokers / N_Smokers

P_stroked_smokers = N_Stroked_smokers / NStroked_patients

print(100 * '-')

# Estudio de la edad
Stroked_ages = df_array[Stroked_patients, 2]
Stroked_smokers_ages = df_array[Stroked_Smokers, 2]
ages = df_array[:, 2]

# Estudio de los niveles de glucosa
Stroked_glucose = df_array[Stroked_patients, 8]
Stroked_smokers_glucose = df_array[Stroked_Smokers, 8]
Smokers_glucose = df_array[Smokers, 8]
Diseased_glucose = df_array[Diseased_patients, 8]
Patients_glucose = df_array[:, 8]

# Estudio de la distribución de glucosa con respecto a la edad

# Si tienes la glucosa alta: Es probable que seas mayor. Lo contrario? NO. Mirar ejemplo siguiente
High_glucose_patients = df_array[:, 8] > 170
High_glucose_patients_ages = df_array[High_glucose_patients, 2]

# Análisis contrario.

Old_patients = df_array[:, 2] > 55
Young_patients = df_array[:, 2] < 55
Old_patients_glucose = df_array[Old_patients, 8]
Young_patients_glucose = df_array[Young_patients, 8]

# Reformating data for a cuantitative analysis

df_array[Males, 1] = 0
df_array[Females, 1] = 1

df_array_cov_matrix = df_array[~bmi_incomplete, :]
selected_atributes = [2, 3, 4, 8, 9,-1]
selected_atributes_name = ['Gender','Age','hypertension','heart disease','glucose level','Stroke']
df_array_cov_matrix = df_array_cov_matrix[:, selected_atributes]




feat_means = np.mean(df_array_cov_matrix, axis=0)
centered_df_array = df_array_cov_matrix - feat_means
std_df_array          = np.std(centered_df_array.astype(float), axis=0)

#We now obtained the NORMALIZED dataset

normalized_df_array = centered_df_array / std_df_array

# The PCA analysis is to be performed on the feature space, not taking into account output information (Stroke)
normalized_df_array_PCA = normalized_df_array[:,:-1]


correlation_matrix = np.cov(normalized_df_array.astype(float).T)
correlation_matrix_PCA = np.cov(normalized_df_array_PCA.astype(float).T)
values, vectors = np.linalg.eig(correlation_matrix_PCA)

eig_order = np.argsort(-values)

sorted_values = values[eig_order]
total_variance = sum(sorted_values)
sorted_vectors = vectors[:,eig_order]





print("***********************************************") # For debuging purposes

#Results representation 

# EigenValues of the feature variables selected for the analysis.
plt.scatter(np.arange(len(sorted_values)),sorted_values)
plt.title("Sorted Eigenvalues of the Covariance matrix")
plt.ylabel("% of total variance")
plt.show()

ax1 = seaborn.heatmap(correlation_matrix,annot = True, linewidth = 0.5)
plt.title('Correlation Matrix',fontsize = 15)
ax1.set_xticks(np.arange(6) + 0.5)
ax1.set_xticklabels(selected_atributes_name, rotation='vertical', fontsize=10)
ax1.set_yticks(np.arange(6) + 0.5)
ax1.set_yticklabels(selected_atributes_name, rotation='horizontal', fontsize=10)
plt.show()

fig, axs = plt.subplots(2, 2)

# First figure
axs[0, 0].hist(Stroked_ages, bins=50, label='Stroked', density=True)
axs[0, 0].hist(Stroked_smokers_ages, bins=50, label='Stroked Smokers', density=True)
axs[0, 0].hist(ages, bins=50, alpha=0.4, label='All patients', density=True)
axs[0, 0].legend(loc='upper right')
axs[0, 0].set_title('Ages histograms')

# Second figure
axs[0, 1].hist(Stroked_glucose, bins=50, alpha=0.4, label='Stroked', density=True)
axs[0, 1].hist(Stroked_smokers_glucose, bins=50, alpha=0.4, label='Stroked Smokers', density=True, )
axs[0, 1].hist(Patients_glucose, bins=50, alpha=0.4, label='All patients', density=True)
axs[0, 1].hist(Diseased_glucose, bins=50, alpha=0.4, label='Diseased', density=True)
axs[0, 1].legend(loc='upper right')
axs[0, 1].set_title('Glucose levels histograms')

# Third figure
axs[1, 0].hist(High_glucose_patients_ages, bins=50, alpha=0.4, label='-', density=True)
axs[1, 0].set_title('High Glucose age distribution')

# Fourth figure
axs[1, 1].hist(Old_patients_glucose, bins=50, alpha=0.4, label='Old patients', density=True)
axs[1, 1].hist(Young_patients_glucose, bins=50, alpha=0.4, label='Young patients', density=True)
axs[1, 1].set_title('Patients Glucose')
axs[1, 1].legend(loc='upper right')

plt.show()
print(100 * '*')















