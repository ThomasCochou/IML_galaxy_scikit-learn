# coding=utf-8

#######################################################
# LOAD DATA
#######################################################

import os,glob
import numpy
import fitsio
from matplotlib import pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# Initialisation d'une liste pour contenir les images
# et d'une liste pour contenir les types morphologiques
raw_data =[]
type_morph =[]

 

# Définir les chemins vers les images et le catalogue,
# mydataDir ='./data/'
mydataDir = 'data/'
mycatalog_path =os.path.join(mydataDir,'data-COSMOS-10000-id.txt')
mypath_template ='data/image/*'
 

# Chargement des données qui nous intéressent dans le catalogue
ids, mod = numpy.loadtxt(mycatalog_path, unpack=True, usecols=(0,3))

# Chargement des images
for one in glob.glob(mypath_template):
   # Extraction de l'id à partir du nom de fichier
   print(one)
   idi =int(one.split('/')[-1].split('_')[0])
   print(idi)
   modi = mod[ids==idi][0]
   print(modi)
   # On va ignorer les out layer ie les mod == 0
   if modi>0:
      # Ajout de l'image
      data = fitsio.read(one)
      raw_data.append(data)
      # Ajout du type morphologique
      type_morph.append(modi)

# Reformatage en numpy pour plus de facilité
raw_data = numpy.asarray(raw_data)
type_morph = numpy.asarray(type_morph)

#######################################################
# display
#######################################################

# Un petit graphique pour illustrer
fig = plt.figure(figsize=(7,7))
for i in range(9):
   plt.subplot(330+i+1)
   plt.imshow(raw_data[i])
   plt.title('type %d'%(type_morph[i]))
   plt.axis('off')
# Visualiser le plot
plt.show()


#######################################################
# DATA TRANSFORMATION
#######################################################

# Normalisation des images
data_scaled = numpy.asarray([(img-img.mean())/img.std()for img in raw_data])

# Transformation en 1d array
data_1d = data_scaled.reshape((data_scaled.shape[0],-1))

# Vérifions avec la dimension des données sous forme de vecteurs
print(data_1d.shape)

# Et une valeur moyenne d'environ 0
print(data_1d.mean())

# Separate data

X_train, X_test, y_train, y_test = train_test_split(data_1d, type_morph, test_size=0.3,random_state=109) # 70% training and 30% test


# SVM

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("~~~ SVM ~~~")

print("confusion_matrix")

print(cm)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("F-score:",f1_score(y_test, y_pred, average='macro'))


# Random Forest

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("~~~ Random Forest ~~~")

print("confusion_matrix")

print(cm)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("F-score:",f1_score(y_test, y_pred, average='macro'))


# Gradiant Boost Classifer

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("~~~ Gradiant Boost Classifer ~~~")

print("confusion_matrix")

print(cm)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("F-score:",f1_score(y_test, y_pred, average='macro'))


# Ada Boost Classifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("~~~ Ada Boost Classifier ~~~")

print("confusion_matrix")

print(cm)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("F-score:",f1_score(y_test, y_pred, average='macro'))


# MLP Classifier

clf = MLPClassifier(random_state=1, max_iter=300) #hidden layer size default 100 pour avoir plusieurs layers hidden_layer_size=(100,10)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("~~~ MLP Classifier ~~~")

print("confusion_matrix")

print(cm)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("F-score:",f1_score(y_test, y_pred, average='macro'))

