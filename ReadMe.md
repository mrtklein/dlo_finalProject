# Abschlussprojekt für das Modul DLO (Stein - Schere - Papier)

## Group 6:
* Tim Keinoth, Mat.-Nr. 11134557
* ...
* ...

Zurzeit nur das Projekt aus den Quellen in die Projektstruktur gebracht und teilweise anpassungen gemacht, um es lauffähig zu bekommen.



## Quellen
* https://www.kaggle.com/code/twhitehurst3/rock-paper-scissors-keras-cnn-99-accuracy

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
* ToDo: Preprocessing der Bilder, da diese unterschiedliche shapes haben


* Ausführen über Google Colab (Projekt muss in GDrive gespeichert sein). 
Datei run_from_colab.ipynb bei Colab hochladen und ausführen.

WARN: Ich konnte das Projekt nicht direkt in GDrive übertragen, da dann Git-Berechtigung gefehlt, um zu committen
