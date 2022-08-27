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

* ToDo:
  * Filter size & Padding & Image size => unterschiedlich große Seitenverhältnisse ML
  * FCL - Size 



Kochbuch vom Salmen:

Step 1:
 * Lossfunktion festlegen
 * Leistung des Netzes testen

Step 2:
 * Andere netze im gleichen Bereich vergleichen
 * Topologische struktur entscheiden (CNN)
 * Aktivierungsfunktion festlegen
 
Step 3:
 * Optimierungsalgorithmus festlegen (SGD mit Momentum/learning rate decay/adam)
 * Batch normalization testen 
 * Regularisierung wenn nötig
 * transfer learning mit vortrainierten Netzen (ggf.)
 
Step 4:
 * Vergleich Trainingsfehler mit Ziel/Erwartung (Folie 394)
 * Schlechte Ergebnisse im Training? -> Netz/Training verändern
 * Gute Ergebnisse im Training aber schlechte auf Testdaten? -> Hyperparameter optimieren

Step 5 (Optimierung):
 * Effektive Kapazität des Modells anpassen an Komplexität des Problems
   -> Mehr Neuronen/Verbindungen
   -> CNN Filter Vergrößern
   -> Dropout verringern


Inhaltsverzeichnis

1. Einleitung {Tobi}
2. Related work (Kaggle) {Tobi}
3. Daten, Dataset Augmentation {Flo}
4. Modell Architektur, Lossfunktion, Aktivierungsfunktion,Baseline tests {Tim}
	- Quellen zu anderen Klassifizierungsnetzen
	- Warum Categorical cross entropy (Flo)
	- Warum ReLU
	- Warum Softmax
5. Leistung Netz Testen, Early stopping {Flo}
	- Baseline tests: Tests mit wenigen bildern, um zu zeigen das das Modell das Problem lernen kann
6. Regularisierung {Flo}
	- Was ist Overfitting (Warum Regularisierung) [kurz]
	- Was wurde gemacht (Methoden erklären [kurz])
7. Optimierung (Fehlt noch)
8. Zusammenfassung und Ausblick (Fazit)

