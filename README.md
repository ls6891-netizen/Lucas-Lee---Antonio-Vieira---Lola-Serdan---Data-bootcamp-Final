# DATABOOTCAMPFINAL
Lucas Lee, Antonio Vieira and Lola Serdan final project

Introduction:

Our group chose to study heart disease because, out of all of the preventable global health causes, cardiovascular disease stood out as something personal to us, with a scientifically urgent rationale. According to the World Health Organisation (WHO), cardiovascular disease is responsible for 17.9 million deaths annually. This daunting number cements heart disease as one of the most common and horrific causes in history. Early detection and prediction of heart disease is crucial for preventative measures, lowering annual death rates. Our project aspires to serve as both a prediction model and a learning approach to understand and forecast the presence of heart disease, using clinical patient data from the UCI Machine Learning Repository.

The predictive model and task focus on binary classification, which aims to determine whether a patient has heart disease (class 1) or does not have heart disease (class 0). Based on 13 clinical features, including age, sex, chest pain type, blood pressure, cholesterol levels, fasting blood sugar, and various other cardiac measurements. These variables provide a broad snapshot of general health indicators, thorough enough to help solve and understand certain heart-specific conditions. To model the relationship between the inputs and the outcome, the neural network is constructed with 5 layers, implementing batch normalisation for increased training speed and dropout regularisation to reduce overfitting, while improving generalization. 

Deep learning approaches from PyTorch demonstrate strong predictive performance on the related medical classification; the test accuracy scored approximately 84-85% and an AUC-ROC score of 0.92. The model dealt with and learned non-linear patterns in complex clinical data, resulting in balanced performance, with 85% sensitivity and 87% specificity. These percentages reflect strong performance and suggest that our neural networks can effectively support cardio disease diagnosis when trained on clinical features, though additional data and model refinements could further improve performance.

Behind the 17.9 million figure is an even greater number of families and loved ones who’ve experienced loss relating to this statistic. When deciding what data to use, we initially thought to choose something financial, like many Stern students in the class; with the amount of data and importance, it felt fitting to base our project on this. On top of the emotional motivation, cardiovascular disease is one of the most well-documented public health challenges on earth. This prevalence spans age groups, gender, geographic location, and physical condition. Working with such a widely studied medical condition gave us more leeway in choosing out dataset. The UCI Heart Disease dataset covers every major feature along with a diverse mix of various other categories, like ECG, chest pain type, and so forth, tying directly to physiological processes. Our data set had plenty to train predictive models; nonetheless, structured for a fully fledged workflow covering preprocessing, imputation, scaling, model training, evaluation, and feature interpretation.




Data Description

Source and Composition:


The dataset used in this analysis is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository, originally collected by the Cleveland Clinic Foundation. This dataset is widely recognized in the machine learning community as a benchmark for cardiovascular disease prediction tasks and has been used in numerous academic studies on medical diagnosis. The data represent real clinical measurements from patients undergoing cardiac evaluation, making it highly relevant for developing practical diagnostic tools.

We aggregated data derived from four clinical centers:

- Cleveland
- Hungarian Institute of Cardiology
- University Hospital Zurich (Switzerland)
- Veterans Affairs Medical Center (VA)
	
We combine these sources into a unified dataset totaling 920 patient records, calling 4 different URLs, containing data from Cleveland, Hungary, Switzerland, and Virginia. Each has 13 features and one target variable. The original target ranges from 0 to 4, describing disease severity. Consistent with much of the ML literature and with our clinical objective, screening for any heart disease, we convert this into binary classification:

-	target = 0 → no disease
-	target > 0 → disease
	
Missing values (represented as “?” in the raw files) are imputed using median imputation, a robust method appropriate for skewed clinical data.


Feature Definitions and Clinical Rationale


The 13 features retained in the analysis capture demographic factors, physiological measurements, stress-test results, and imaging indicators. Each is clinically relevant:

- Age: Cardiovascular risk rises sharply with age; a key baseline predictor
  
- Sex: Men have historically higher risk
  
- Chest Pain Type (cp): One of the most direct signals of coronary artery disease
  
- Resting Blood Pressure (trestbps): Hypertension is a known risk factor for cardiac events
  
- Serum Cholesterol (chol): Elevated cholesterol contributes to atherosclerosis
  
- Fasting Blood Sugar (fbs): Reflects diabetic status, a major comorbidity influence cardiovascular risk
  
- Resting ECG (restecg): Detects baseline rhythm abnormalities
  
- Maximum Heart Rate (thalach): Low peak heart rate may indicate impaired cardiac function
  
- Exercise-Induced Angina (exang): A strong clinical marker of ischemic disease
  
- ST Depression (oldpeak): A quantitative ECG measure associated with myocardial ischemia
  
- Slope of ST (slope): Number of Major Vessels (ca): Imaging data indicating coronary obstruction; highly predictive
  
- Thalassemia / Thallium Test (thal): Indicates blood-flow abnormalities or anemia affecting oxygen delivery


We chose these features because each is a good representation of distinct physiological dimensions known to be relevant indicators of cardiac risk. These parameters also go well with Neural networks, as they are especially effective at learning interactions among such heterogeneous predictors.



Initial Data Analysis


Before modeling, several patterns emerge:

- Class balance: The dataset is approximately balanced between disease and non-disease cases. This allows accuracy to be a meaningful metric without the need for aggressive resampling.
  
- Correlations: The strongest correlations with disease include chest pain type, ST depression, Thalassemia, exercise-induced angina, and maximum heart rate. These align closely with established diagnostic criteria in cardiology.
  
- Age and Cholesterol Distributions: Individuals with disease tend to be slightly older, and cholesterol distributions are more diffuse among diseased patients. Yet substantial overlap exists, illustrating why simple threshold rules perform poorly.
  
- Age vs. Cholesterol scatter: No clear linear boundary separates the classes, suggesting that linear models are insufficient and justifying the use of flexible non-linear techniques such as neural networks.

Data Preprocessing 


All features were standardized with scikit-learn’s StandardScaler, which rescales each variable to have zero mean and unit variance. This step is important for neural networks because it ensures that no feature dominates the learning process simply due to its scale. 
The dataset was then split into training (80%) and test (20%) sets using stratified sampling to preserve the original class balance. This helps ensure that both sets reflect the overall population.


Our NN Architecture


We chose to work with a core model that is a five-layer feedforward neural network (multi-layer perceptron) designed to learn complex nonlinear relationships between clinical features and heart disease presence.


Network Structure:


- Input layer: 13 features (clinical measurements)
- Hidden layer 1: 128 neurons with ReLU activation and batch normalization
- Hidden layer 2: 64 neurons with ReLU activation and batch normalization
- Hidden layer 3: 32 neurons with ReLU activation and batch normalization
- Hidden layer 4: 16 neurons with ReLU activation
- Output layer: 2 neurons (binary classification with softmax)
  
Our chosen structure uses a pyramidal structure (commonly used) with layers that get smaller as the data move through the model. This design helps the network take in a lot of information at first and then gradually narrow it down to the patterns that matter most.


Key Architectural Components


These were critical implementations that helped us improve our final accuracy and precision, as well as preventing overfitting 


ReLU Activation Functions: We use ReLU as the activation function in our hidden layers. It is the most friendly approach to neural networks in Python because it is efficient, fast to compute and helps the model learn non-linear patterns.


Batch Normalization: This step is included after each hidden layer to help the network train more smoothly. It keeps the layer outputs on a similar scale, which makes learning fasster and prevents the model from getting stuck early in training.


Dropout Regularization: We applied a dropout rate of 0.3, meaning that during training about 30% of the neurons in a layer are randomly turned off. This prevents the model from depending too heavily on specific neurons and was helpful in reducing overfitting. Dropout is especially useful given our relatively small dataset.


Training Methodology


We built and trained our model using PyTorch, mainly because it is one of the most common and easy-to-use frameworks for neural networks. PyTorch handles things like gradients and tensor operations for us, so we can focus on designing the model rather than worrying about the math behind it.

For optimization, we used the Adam optimizer with a learning rate of 0.001. 

Loss Function: We used CrossEntropyLoss, which is the usual loss function for classification problems. It compares the model’s predicted probabilities to the true labels and pushes the model to give higher confidence to the correct class. It is stable, widely used, and works well with softmax outputs.


Training Process


During training, we ran the model for up to 100 epochs using mini-batches of 32 samples. Each training step follows the standard routine:
1. The model makes predictions (forward pass)
2. We compute the loss
3. PyTorch calculates the gradients (backward pass)
4. Adam updates the weights
Using mini-batches instead of one example at a time makes training faster and gives more reliable gradients.


Monitoring and Early Stopping:


Early stopping was a critical addition. We tracked the training and test loss at each epoch to make sure the model was learning properly. The loss curves were smooth and consistently decreased, which showed that the learning rate and batch size were reasonable. We also used early stopping so the model would stop training once it stopped improving, which helps prevent overfitting. In practice, our model stopped training at 42 epochs.
Overall, our training setup follows the typical and recommended approach for building a basic neural network. It uses well-established defaults—Adam, CrossEntropyLoss, mini-batches, and early stopping—which together give stable and reliable training performance.

Chosen Evaluation Metrics

Model performance is assessed using multiple complementary metrics to provide a comprehensive view of predictive capability:

Accuracy: The proportion of correct predictions, serving as the primary performance metric given the balanced class distribution.

Precision: The proportion of predicted positive cases that are actually correct. Precision provides insight into how reliably the model identifies disease cases without generating excessive false alarms.

Macro Precision: The average precision across both classes, giving equal weight to “No Disease” and “Disease.” This metric ensures that performance is not biased toward one class, making it especially important in medical contexts where errors in either direction carry clinical consequences.

Confusion Matrix: A 2×2 matrix showing true positives, true negatives, false positives, and false negatives, providing insight into the types of errors the model makes.

Classification Report: Precision, recall, and F1-score for each class, offering detailed performance breakdown beyond simple accuracy.

ROC Curve and AUC: The Receiver Operating Characteristic curve plots true positive rate against false positive rate across all classification thresholds. The Area Under the Curve (AUC) summarizes this into a single metric, with 1.0 representing perfect classification and 0.5 representing random guessing.
Feature Importance: Analysis of first-layer weights provides approximate feature importance scores, identifying which clinical measurements contribute most strongly to predictions.

Conclusion:


In this project, we developed and completed an end-to-end pipeline for predicting heart disease based on multiple UCI repositories using genuine clinical data. Though modest, our dataset included every important metric and covered the bases to form an accurate prediction. While the performance metrics are encouraging, this project underscores the limitations seen in classical medical datasets. Our model was trained on fewer than 1,000 patient records; this figure sits far below the size typically required for clinical deployment. On top of this, the features we used were constrained to numerical variables, whereas in real clinical decision-making, you often’ll see forms of imaging, notes from physicians, historical data, and other nuanced biomarkers. We hope this model can serve as an academic demonstration of a predictive model with the potential for future real-life use and significance. 

The final results for our model’s performance were:

- Final Test Precision: 0.84
- Final Train Precision: 0.84
- Macro Precision (weighed-average): 0.85
- AUC-Score: 0.91


One of the main insights from our project was identifying which features were most useful for predicting a patient’s diagnosis. The top five were chest pain, ST depression, number of major vessels, maximum heart rate achieved, and blood oxygen capacity. These are well-known indicators in medical research, but seeing them emerge from our own model helped us better understand which factors matter most for assessing heart health.

Next Steps:


Future work could focus on making the model more foolproof and useful for real clinical settings. One clear improvement would be to train and test the model on larger and more diverse datasets, either by combining data from multiple sources or using larger public cardiac datasets, which would help ensure the results generalize beyond a single sample. 


Using cross-validation and more structured hyperparameter tuning could also lead to more dependable performance estimates and incremental gains in accuracy. In addition, exploring slightly more advanced model designs or simple ensemble approaches may help capture patterns the current architecture misses. 


From a practical standpoint, adding clearer interpretability and some measure of prediction uncertainty would make the model’s outputs easier to trust and evaluate, especially in borderline cases. Extending the task from binary classification to predicting disease severity and testing the model on fully independent datasets would further increase its clinical relevance. 


Overall, this project shows that deep learning can perform well on cardiac classification problems and provides a substantial starting point for future improvements aimed at real-world use. The combination of strong predictive performance, interpretable feature importance, and rigorous evaluation methodology establishes a solid baseline for future enhancements aimed at clinical deployment.


