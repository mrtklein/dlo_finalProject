# Abschlussprojekt für das Modul DLO (Stein - Schere - Papier)

## Group 6:
* Tim Keinoth, Mat.-Nr. 11134557
* Tobias Giesler, Mat.-Nr. 11114022
* Florian Graf, Mat.-Nr. 11129196



## Quellen
* https://www.kaggle.com/code/twhitehurst3/rock-paper-scissors-keras-cnn-99-accuracy (Erste Vorlage)
* https://www.kaggle.com/code/quadeer15sh/tf-keras-cnn-99-accuracy
  * (Flo schau dir mal Abschnitt "Dividing the Images and Labels" an), (Evtl auch nutzbar sklearn.model_selection.train_test_split)
  * (Tobi Abschnitt "Visualizing the Images") hab die Visualisierung erstellt, würde sehr gerne morgen mit dir (Tim) über die Filter quatschen.
* https://github.com/SouravJohar/rock-paper-scissors => Inder von Youtube (https://youtu.be/0uSA3xyXlwM) 

* Kochbuch ML: https://1drv.ms/u/s!AjNqP96LkzdkhrphpBshiO2RImfU0g?e=pqHTzW 
  * Kapitel 8: Bilder verarbeiten (z.B. Filter etc.)


## Einrichtung

* Über die Anaconda Prompt eine Umgebung erstellen und alle nötigen Packagen installieren
```python
conda create --name dloEnv --file requirements.txt
```

* File | Settings | Project: rock_paper_scissors | Python Interpreter => Interpreter auswählen


* Ausführen über Google Colab:
  * script run_py_from_drive.ipynb bei colab hochladen
  * falls das Repo noch nicht auf hdrive ist muss der input ordner noch in das repo kopiert werden



Aufgaben verteilung:
* ToDo: Datenvisualisierung 
--> Bilder in Ordner speichern

* ToDo: Optimierung
  * ImageDataGenerator => Regularization
  * Callbacks fertig implementieren
  * GridSearchCV einbauen





