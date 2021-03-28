# Automated Framework for Diabetic Retinopathy Diagnosis using Deep Learning
The projects aims to develop an automated system using Deep Learning to classify the retinal fundus images into different stages of Diabetic Retinopathy

### Abstract

* Diabetic Retinopathy (DR) is one of the common retinal diseases leading to blindness
* In the current clinical diagnosis, the detection mainly relies on the experienced ophthalmologist examining the color fundus image and then evaluates the patient’s condition which is a tedious and time consuming task
* This project is an attempt to provide an automatic diagnosis of DR by performing fundus image classification using CNN

### Dataset Description

The EyePacs dataset was used for image classification. The dataset description is as given below:

![datades](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/dataset_description.png)</br>

### Steps Involved in Deep Learning Ensemble Classifier Framework

#### Preprocessing 
* Cropping
* Normal Bilinear Rescaling
* Filtering Black Images

![preprocess](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/preprocess.png)</br>

#### Training:
* Feature Extraction using VGG16 and Inception v3 networks
* Training the fully connected layers with SMOTE Oversampling
* Saving the trained models for prediction

The sequence of steps used for training the model is illustrated in the flowchart below:

![trainflowchart](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/training_flowchart.png)</br>

#### Testing:
* Prediction of images using the trained models
* Using the ensemble algorithm for prediction


### Results Section

The confusion matrix for the Deep Learning Ensemble Classifier is given below:

![confmatrix](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/confusion_matrix.png)</br>

The overall test accuracy was found to be **78.04 %**

The values of the other statistical metrics for every class is stated below:

![other_metrics](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/other_metrics.png)</br>

Kappa Value =  0.57272 <br/>
Area under curve = 0.72 <br/>
Misclassification or error rate = 0.22                                  
Null error rate = 0.26 <br/>

The graph of model accuracy versus number of epochs is found below:

![acc](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/model_loss.png)</br>

The graph of model loss versus number of epochs is found below:

![loss](https://github.com/sumanth-bmsce/Diabetic-Retinopathy-Image-Classification/blob/master/model_accuracy.png)</br>

### Conclusion

* The proposed ensemble model using VGG16 and Inception v3 networks could achieve an overall accuracy of 78.04 % on the test dataset while individually VGG16 network and Inception v3 network produced accuracy of 66.20% and 61.32% respectively. 

* In this study, we have also dealt with the imbalanced dataset problem using oversampling with SMOTE to prevent overfitting.
 
* The accuracy of our proposed ensemble model is also compared with other state of the art classifier and it was observed that our model performance was the best

### Future Work

* There is still scope in improving the model performance by varying the different hyperparameters and coming up with a better ensemble technique.

* The product can be modified by adding other parameters such as age, haemoglobin AIC values, genetic factors, duration of diabetes and patient history can be added along with the image dataset to increase the sensitivity metric of the proposed system.

* The product capabilities can be enhanced by adding other ophthalmologic disease diagnosis modules with existing Diabetic Retinopathy module

### References

[1] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, Gradient-based learning applied to document recognition, Proceedings of the IEEE (1998), Vol. 86, Pages: 2278-2324.

[2] Arkadiusz Kwasigroch, Bartlomiej Jarzembinski and Michal Grochowski, Deep CNN based decision support system for detection and assessing the stage of diabetic retinopathy, IEEE International Interdisciplinary PhD Workshop (2018).

[3] Gargeya  R, Leng  T, Automated identification of diabetic retinopathy using deep learning.  Ophthalmology (2017); Vol. 124(7), Pages: 962-969.

[4] Harry Pratt, Frans Coenen, Deborah M.Broadbent, Simon P.Harding, Yalin Zheng, ‘Convolutional Neural Networks for Diabetic Retinopathy’, Procedia Computer Science (2016), Vol. 90, Pages: 200-205.

[5] P. Roy, R Tennakoon, K. Cao, S. Sedai, D. Mahapatra, S Maetschke, R. Garnavi, ‘A novel hybrid approach for severity assessment of diabetic retinopathy in colour fundus images’,IEEE International Symposium on Biomedical Imaging (2017).

[6] Shaohua Wan, Yan Liang, Yin Zhang,’Deep convolutional neural network for diabetic retinopathy detection by image classification’, Elsevier Computers and Electrical Engineering (2018), Vol.72, Pages: 274-282.

[7] Karan Bhatia, Shikhar Arora, Ravi Tomar,Diagnosis of Diabetic Retinopathy Using Machine Learning Classification Algorithm’, IEEE 2nd International Conference on Next Generation Computing Technologies 2016.

### Contributors

1. Sumanth Simha C
2. Nagaraj G
3. Harish Chandra G R
