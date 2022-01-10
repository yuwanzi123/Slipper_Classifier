# Slipper_Classifier
This is a personal project to train a Logistic Regression Model to classify if an image shows slippers. 

The "database" folder contains: trainning dataset, validation dataset, test dataset and ground truth dataset. The trainning dataset contains 9320 pictures; validation dataset has 996 pictures. 

The "python_script" folder contains some supporting code and the main code named "slippers_a_classifications.py"

I designed two versions of this model. The first version has a Training database with 1309 pictures and a Testing database with 141 pictures. The best performance of this version is Training accuracy:68%; Testing accuracy: 65%. 

In the second version, I tried to increase the accuracy by increasing the database size. So I used data augmentation theories. The second version has a Training database with 9,320 pictures and a Validation set with 996 pictures and a Testing database with 1,000 pictures. The best performaence with this version: Train accuracy: 57%; Test accuracy: 53%. 


For detaied information, please visit this blog: https://slipperclassifier.blogspot.com/
