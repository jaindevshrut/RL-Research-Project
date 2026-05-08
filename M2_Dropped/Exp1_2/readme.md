# **Project Overview**

## **Sequence of Execution**


### **1. Determining the Classifier**
The following steps are dedicated to analyzing human data, generating training and testing data, and determining the classifiers.


#### **1.1 Analysis of Human Data**
Run `01_classifier/01_analyze_human_data.py` to analyze human data, providing the **mean and variance** of human emotion precision.

Human data can be accessed with

- `data/human_free.csv`
- `data/human_limit.csv`
#### **1.2. Data Generation**

Run `01_classifier/02_classifier.py` to generate training and testing data using Scherer's table. The resulting files are:

- `data/classifier_test.csv`
- `data/classifier_train.csv`


#### **1.3. Classifier Determination**
Execute `01_classifier/03_determine_classifier_c.py` to determine the optimal C values for free and limit classifiers, aligning closely with human behavior. The classifiers are:

Human Free (loose classifier): c = 0.0032, variance = 0.0002  
Human Limit (restricted classifier): c = 0.014, variance = 0.0056

### **2. MDP Model Execution**
Run `02_mdp_model/01_get_model_data.py` to run the MDP models and obtain the appraisal vectors for each emotion.The results are saved in:

- `data/model_result.csv`

### **3. Model Emotion Prediction**
Utilize `03_model_infer/01_svm_infer.py` to predict emotions for model data using the classifiers from step 3 and model data from step 4. The results will be saved in:

- `data/svm_free_0.0032_var.csv`
- `data/svm_limit_0.014_var.csv`

### **4. Statistical Analysis**
Generate plots for Figures 4 and 5 by running:

- `04_statistical_analysis/Exp1_free_analyse.R`
- `04_statistical_analysis/Exp2_limit_analyse.R`