Is machine learning scary/intimidiating ?

Lets implement, according to the internet, the simplest machine learning algorithm, k-Nearest Neighbors.

<h4>problem description</h4>
The k-Nearest Neighbors algorithm (kNN) is a machine learning algorithm used for both classification and regression. The algorithm works by calculating the distances of a given test item to the rest of the dataset and uses a majority vote to classify the test item. The goal of this project is to run the kNN algorithm on the MNIST dataset of handwritten numbers, and determine its accuracy.
<br>

<h4>kNN with k = 1</h4>

The first step for implementing kNN is to generate the distance arrays for each of the 10,000 testing images tested against the 60,000 training images. To accomplish this, we will calculate the Euclidian distance between a given training image against all of the testing images, and save the 10 lowest distances associated with that test images actual value. This operation will execute 10,000 x 60,000 operations. We save the lowest 10 so we can save and resuse the distance arrays in future calculations with k's greater than 1.

The Euclidian distance function utlizes the numpy library to allow for a more efficient vectorized calculation. For given matrices a and b we will calculate the distance as `sqrt((sum(a-b)^2))`.

After we generate the distances for each image, we can then loop through each image's distance array, and grab the lowest distance for each. The image associated with the lowest distance represents the predicted value of the training image.  The psuedocode for the kNN algorithm is given below:
```
knn {
  Generate distances array
  Grab k lowest distances
  Perform majority vote on actual image value associated with each distance
  Return the winner of the majority vote
}
```

<b>NOTE</b>: In case of a tie in the majority vote, the first max occurance in the k lowest distances is returned as the prediction.

Source code for implementation in kNN.py

<h4>10-fold cross validation to find optimal k</h4>

What happens if you change k? Is a paticular k more accurate ? No idea, thats why someone came up with this idea called cross fold validation.  Cross fold validation aims to find an optimal k. The first step is to create 10 folds with each being created from the training set images. The first fold will use images 1 to 6000 as its training set and use images 6001 to 60000 as its testing set, the second fold will use images 6001 to 12000 as its training set and use images 0 to 6000 and images 12001 to 6000, etc etc. Next, we will generate the distances array for each fold (6000 x 54000), saving the lowest ten distances as we did in the previous step. Once generated we will run kNN with k's ranging from 1 to 10 and count the number of correct predictions. Once caluclated we can then calculate the accuracy of each k and find our optimal k. The psuedocode for this process is given below:

```
corss_validation {
  Generate folds
  Generate distances array for each fold (1-10)
  Run the kNN algorithm on each fold with k's ranging from 1 - 10
  Calculate accuracy based of off correct predictions with each k accross all folds
}
```

The runtime for this process is incredibly long. Running the process on my 4 year old laptop took approximatly 18 hours and I was able to cook some eggs on it. Nice.
I tried looking into using an AWS instance but got annoyed.

Results

oh these results are really wrong jesus what happened.
I found out what happened. The images gathered from keras and subsequently stored were stored as integers. This really messed up calculating the distances.
Add these two lines to cast the images to floats.
I now have to run this again for 18 hours. Oh boy.

```
Total correct predictiosn for each k throughout all 10 folds.
[16399, 16399, 16394, 16390, 16389, 16379, 16366, 16341, 16324, 16302]

Accuracies
[0.27332, 0.27332, 0.27323, 0.27317, 0.27315, 0.27298, 0.27277, 0.27235, 0.27207, 0.27170]
```

With this, a k = 1 or k = 2 can be used as our optimal k.

Source code for implementation in crossfold.py

<b>kNN with optimal k results</b>

Using k = 2

Total Correct Predictions: 16589
Accuracy = 0.27648

stddev = sqrt((.27*.63)/60000) = .0017

Confidence Interval = [0.287, 0.379]

Confusion Matrix

|  	| 0 	| 1 	| 2 	| 3 	| 4 	| 5 	| 6 	| 7 	| 8 	| 9 	|
|------	|------	|------	|------	|------	|------	|------	|------	|------	|------	|------	|
| 0 	| 1915 	| 6 	| 291 	| 357 	| 207 	| 373 	| 406 	| 136 	| 337 	| 195 	|
| 1 	| 1830 	| 6237 	| 2643 	| 2627 	| 2460 	| 2372 	| 2503 	| 2506 	| 2613 	| 2188 	|
| 2 	| 205 	| 54 	| 995 	| 300 	| 101 	| 176 	| 249 	| 63 	| 306 	| 82 	|
| 3 	| 204 	| 15 	| 225 	| 844 	| 125 	| 308 	| 154 	| 104 	| 300 	| 168 	|
| 4 	| 217 	| 9 	| 182 	| 153 	| 717 	| 186 	| 276 	| 199 	| 163 	| 403 	|
| 5 	| 434 	| 47 	| 296 	| 581 	| 352 	| 828 	| 257 	| 245 	| 537 	| 334 	|
| 6 	| 364 	| 47 	| 421 	| 216 	| 370 	| 287 	| 1417 	| 138 	| 319 	| 249 	|
| 7 	| 499 	| 299 	| 735 	| 836 	| 1136 	| 687 	| 494 	| 2669 	| 747 	| 1676 	|
| 8 	| 169 	| 11 	| 109 	| 132 	| 113 	| 93 	| 74 	| 80 	| 412 	| 99 	|
| 9 	| 86 	| 17 	| 61 	| 85 	| 261 	| 111 	| 88 	| 125 	| 117 	| 555 	|



Conclucsion:
Machine learning is a little less intimidating to me. Andrew Ng is my hero. I need a new laptop. 
