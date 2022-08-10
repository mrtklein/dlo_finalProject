# Abschlussprojekt für das Modul DLO (Stein - Schere - Papier)

## Group 6:
* Tim Keinoth, Mat.-Nr. 11134557
* Tobias Giesler, Mat.-Nr. 11114022
* Florian Graf, Mat.-Nr. 11129196

Zurzeit nur das Projekt aus den Quellen in die Projektstruktur gebracht und teilweise anpassungen gemacht, um es lauffähig zu bekommen.



## Quellen
* https://www.kaggle.com/code/twhitehurst3/rock-paper-scissors-keras-cnn-99-accuracy (Erste Vorlage)
* https://www.kaggle.com/code/quadeer15sh/tf-keras-cnn-99-accuracy
  * (Flo schau dir mal Abschnitt "Dividing the Images and Labels" an), (Evtl auch nutzbar sklearn.model_selection.train_test_split)
  * (Tobi Abschnitt "Visualizing the Images")
* https://github.com/SouravJohar/rock-paper-scissors => Inder von Youtube (https://youtu.be/0uSA3xyXlwM) 

* Kochbuch ML: https://1drv.ms/u/s!AjNqP96LkzdkhrphpBshiO2RImfU0g?e=pqHTzW 
  * Kapitel 8: Bilder verarbeiten (z.B. Filter etc.)

## Einrichtung

* Über die Anaconda Prompt eine Umgebung erstellen und alle nötigen Packagen installieren
```python
conda create --name dloEnv --file requirements.txt
```

* File | Settings | Project: rock_paper_scissors | Python Interpreter => Interpreter auswählen

* Zurzeit funktionieren nur die Bilder von https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors
* ToDo: Bilder, die Herr Salmen rumgeschickt hat wie folgt bereitsstellen:
rock_paper_scissors/input/paper
rock_paper_scissors/input/rest
rock_paper_scissors/input/rock
rock_paper_scissors/input/scissors
* ToDo: DataLoading (splitten in Train, Test und Val) korrigieren (Flo)
* ToDo: Model richtig erstellen (Tim)
* ToDo: Filter für Bilder erstellen (Tobi / bisschen Tim)
* ToDo: Datenvisualisierung (Rohdaten (Bild)+ bearbeitetes Bild (Filter) anzeigen lassen) (Tobi)
--> Bilder in Ordner speichern
--> Nächstes Meeting: Fr. 12.08 10:00 Uhr

* ToDo: Recherche verwandten Projekte, wenn was Gutes gefunden in Readme packen --> (alle)


* Ausführen über Google Colab (Projekt muss in GDrive gespeichert sein). 
Datei run_from_colab.ipynb bei Colab hochladen und ausführen.

WARN: Ich konnte das Projekt nicht direkt in GDrive übertragen, da dann Git-Berechtigung gefehlt, um zu committen
