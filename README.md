# MachineLearning-GlassDataIdentification

The following project rerpesents my first Machine Learning project - a classifying task. 
It was done using the Python programming language.
The algorithm used is MLP - multilayer perceptron.

The data base and the problem that this project is solving:
  *The data base is called "Glass Identification Data Set" and it is proived by the USA Forensic Science Service
  *In this data base the stored information is about 7 types of glass, diferentiated by their content of oxide
  *The ideea is that, the glass found at a crime spot can be used as a clue in solving a case, if correctly identified

The structure of the data base:
  *A matrix composed of 214 rows and 11 columns
  * First column deals with the ordering of the data
  * The rest of the columns: informations about the glass, such as refraction index, or the percentage of the oxide

Data splitting:
  *Training set: 159 samples - 75% of the data
  *Testing set: 54 samples - 25% of the data

Libraries used:
  *Pandas: reading data from ".txt" and ".csv" type files
  *Scikit-learn: for the ML Algorithm

Conclusion:
  *The algorithm is able to make a correct classification based on the provided parameters
  *The accuray obtained is 60% - due to the fact that there was no cleansing done on the data 
