# CUDA-Messung auf der RTX 5070 Ti

Stand: 26. Oktober 2026. Die Messdaten und vorbereiteten Audio-Caches liegen unter `data/benchmarks/cuda` bzw. `data/cache`. Das vorbereitete, noch nicht ausgeführte Reset-Skript entfernt diese Rohdaten; die dokumentierten Ergebnisse bleiben erhalten. Keine zusätzlichen Pakete oder Modellgewichte heruntergeladen.

## Ausführung

- Optimierte PyTorch SDPA-Attention statt expliziter Attention-Matrizen auf CUDA; gleiche Padding- und Kausalmasken.
- BF16 im Vorwärtslauf. Float32 für Mastergewichte, Optimiererzustand, alle zwölf Verluste, Validierung und gespeicherte Modelle. TF32 deaktiviert.
- Fused AdamW, vektorisierte Bewegungsziele, ein vorbereiteter Folgebatch in gepinntem RAM, nicht blockierende Transfers und auf der GPU gesammelte Verluststatistiken.
- CUDA Graphs für Vorwärts-/Rückwärtslauf bei einem Batch pro Update. Lernrate, Clipping, Endlichkeitsprüfung und Optimierer bleiben dynamisch. Abweichend große letzte Batches und Gradient Accumulation laufen ohne Graph. Graphs sind in der UI abschaltbar.
- CPU/DirectML behalten Float32, explizite Attention und den kompatiblen Optimierer. Die Startprüfung testet den jeweiligen Rechenpfad mit Loss, Rückwärtslauf und Update.

Die Modellgröße, Audio-/Objektkontexte, Lernziele und Generierungsregeln wurden für diese Optimierung nicht reduziert.

## Messverfahren

Hardware: RTX 5070 Ti, 16 GiB VRAM, Compute Capability 12.0. Lokal vorhandenes PyTorch `2.8.0.dev20250314+cu128`, CUDA 12.8, Windows, vier PyTorch-CPU-Threads. Setup und Docker pinnen regulär PyTorch 2.8.0; der Container wurde nicht gebaut oder gemessen. Seine Werte können abweichen.

Standardmodell mit 11.316.079 Parametern. Aus dem echten lokalen Katalog wurden reproduzierbar 32 unterschiedliche Songs ausgewählt: 45–150 Sekunden, 120–900 Objekte, 1–8 Objekte/s. Insgesamt 8.734 Objekte, keine beim Vorbereiten ausgeschlossenen Maps. Split: 24 Training, fünf Validierung, drei unbenutzte Testsongs. Das ist ein begrenzter Regressionstest, kein großes Qualitätstraining und keine repräsentative Genre-Stichprobe.

Referenz: ursprüngliche Float32-Attention, synchroner Datentransfer, ursprüngliche skalare Bewegungszielberechnung, AdamW ohne Fused-Pfad und einzelne CPU-Verlustauslesungen. Die alte Batch-Aufbereitung ist für Reproduzierbarkeit in `scripts/benchmarks/reference_batches.py` eingefroren. Alle Varianten verwenden dieselbe Architektur und dieselben Gewichte zu Beginn. Kleine konstante Tensoren der Bewegungsrechnung werden in allen Varianten wiederverwendet; auch die Referenz erhält diese Nebenoptimierung. Die Referenz ist damit ein kontrollierter Vergleich des ursprünglichen Rechenpfads unter striktem Float32, kein unverändertes historisches Binärprogramm.

Durchsatz: fünf Updates aufwärmen, anschließend 150 Updates messen; drei Wiederholungen mit wechselnder Reihenfolge. Gemessen werden Datenaufbereitung, Transfer, Vorwärts-/Rückwärtslauf, Verluststatistik, Gradient-Clipping und Optimierer. Synchronisation vor/nach dem Zeitfenster. Ohne Audio-Erstaufbereitung, Validierung, Dateicheckpoints und HTTP-Fortschritt. Ein Beispiel ist ein Audio-/Objektfenster, kein vollständiger Song. Tatsächliche Beispielanzahl berücksichtigt kurze letzte Batches.

Speicher: PyTorch-Spitzenwerte für aktiv belegten und vom Allocator reservierten Speicher getrennt, einschließlich zusätzlicher Spitzen während der Vorbereitung. Reservierter Speicher enthält aktive Daten und freie Poolbereiche; die Werte dürfen nicht addiert werden. `nvidia-smi` erfasst zusätzlich alle 200 ms GPU-Auslastung und gesamten Gerätespeicher einschließlich Desktop/Treiber/anderer Prozesse. Auslastung ist ein Zeitindikator, kein Anteil der theoretischen Rechenleistung.

### Gemessener Durchsatz und Speicher

Median des Durchsatzes und der je Lauf gemittelten GPU-Auslastung; Speicherwerte sind die höchsten Spitzen aus den drei Läufen inklusive Vorbereitung:

| Variante | Audiofenster/s | GPU-Auslastung ca. | PyTorch aktiv, GiB | PyTorch reserviert, GiB |
| --- | ---: | ---: | ---: | ---: |
| Referenz, Float32, Batch 2 × 4 | 19,1 | 23 % | 0,81 | 0,89 |
| Optimiert, Float32, Batch 2 × 4 | 27,3 | 27 % | 0,40 | 0,46 |
| BF16 + Graph, Batch 8 × 1 | **330,9** | **77 %** | 0,63 | 0,99 |
| BF16 + Graph, Batch 32 × 1 | 450,2 | 79 % | 2,17 | 4,30 |

Empfohlener Einstieg: **Standardmodell, CUDA, Batch 8, Updates sammeln 1, automatische Genauigkeit, CUDA Graphs an, Lernrate 0,0003.** Dieser Pfad erreicht das **17,4-Fache** des Referenzdurchsatzes bei gleicher nomineller effektiver Batch-Größe. Er wurde auch im Qualitätsvergleich geprüft. Die drei Durchsatzwerte waren 329,5 / 360,1 / 330,9 Fenster/s; Referenz 19,05 / 18,75 / 19,07.

Batch 32 erreicht das 23,6-Fache der Referenz, verändert aber die effektive Batch-Größe auf 32. **Für diese Einstellung wurde nur Durchsatz/Speicher, keine Lernkurven-Gleichwertigkeit geprüft.** Nicht automatisch auf den größeren Wert umstellen, nur um VRAM zu füllen. Je nach Datensatz und Modellgröße kann ein anderes Verhältnis günstiger sein.

Während des aufgewärmten Batch-8-Laufs waren höchstens 0,37 GiB aktiv; die 0,63 GiB in der Tabelle stammen aus der Vorbereitung. Gesamter Gerätespeicher laut `nvidia-smi`: Spitzen 3,20 GiB Referenz, 3,17 GiB Batch 8 und 6,48 GiB Batch 32. Dies enthält weiteren Speicher außerhalb des PyTorch-Allocators und ist kein isolierter Modellwert.

Die beschleunigten 150 Updates dauerten bei Batch 8 nur 3,25–3,55 Sekunden: 16–18 Auslastungsstichproben je Lauf. Batch 32 dauerte etwa zehn Sekunden mit 50–51 Stichproben. Deshalb die Auslastungsprozente als grobe Orientierung lesen, nicht als präzisen Dauerlastwert. Der erste Vorwärts-/Rückwärtslauf eines frischen Prozesses benötigte in der Referenz etwa 18 Sekunden inklusive Initialisierung; spätere Graph-Vorbereitung lag in diesem Prozess unter einer Sekunde. Im echten Backend dauerte die erste kurze Trainingsepoche einschließlich erstmaliger Kernelvorbereitung rund 17 Sekunden. Der ausgewiesene Geschwindigkeitsfaktor gilt für den aufgewärmten Trainingsabschnitt, nicht für den gesamten Ablauf inklusive Audioaufbereitung, Validierung, freier Generierung und Speicherung.

## Qualitätsvergleich

Vor Beginn festgelegt: Seeds 17, 42, 73; je Variante 120 Optimiererupdates mit acht Beispielen pro Update. Referenz und Varianten ohne Graph verwenden Batch 2 mit vier Akkumulationen, Graph Batch 8 mit einer. Die Beispielreihenfolge und Augmentierung sind gleich; unvollständige Gruppen werden für diesen Vergleich über Epochengrenzen aufgefüllt. Aufgrund von Dropout, Rundung und Masken-Normalisierung ist die Gradientenfolge bei geänderter Batch-Aufteilung nicht bitgleich.

Vier Varianten: ursprünglicher Pfad, optimierter Float32-Pfad, BF16 ohne Graph, BF16 mit Graph. Alle werden mit **derselben ursprünglichen Float32-Attention** auf denselben fünf Prüfsongs bewertet. Kein Testset-Tuning. Bewertungszeitpunkte: Update 40, 80, 120; Vergleich der Mittelwerte bei Update 120. Zusätzlich freie Generierung von bis zu 128 Objekten eines Prüfsongs pro Lauf.

Vorab gesetzte Regressionsgrenzen relativ zur Referenz: mittlerer Prüfverlust höchstens `max(0,03; 2 % des absoluten Referenzverlusts)` schlechter; Timing-F1 höchstens 0,02 niedriger; Bewegungsfehler höchstens 5 % höher. Keine statistische Äquivalenzbehauptung: drei Seeds und ein kleiner Korpus können seltene oder langfristige Probleme nicht ausschließen.

Numerische Prüfung bei deaktiviertem Dropout: Float32-SDPA und Referenz haben praktisch identische Ausgaben/Gradienten; BF16 darf eine kleine numerische Abweichung haben. Zusätzliche Tests prüfen Padding/Kausalität, wechselnde Graph-Eingaben, aktualisierte Gewichte, korrekt ersetzte Gradienten, wechselnden Dropout, Prefetch-Fehler, Nicht-Endlichkeitsabbruch und die Fortsetzung des Fused-Optimierers.

### Gemessene Qualität

Mittelwerte über drei Seeds nach jeweils 120 Updates, niedrigere Fehler/Verluste sind besser:

| Variante | Prüfverlust | Timing-F1 | Bewegungsfehler | Objekttyp korrekt |
| --- | ---: | ---: | ---: | ---: |
| Ursprünglicher Pfad | 1,98385 | 58,74 % | 0,26643 | 57,87 % |
| Optimiert, Float32 | 1,98984 | 58,80 % | 0,26696 | 59,96 % |
| BF16 ohne Graph, Batch 2 × 4 | 1,97918 | 59,73 % | 0,26541 | 60,04 % |
| BF16 mit Graph, Batch 8 × 1 | 1,98629 | 60,08 % | 0,26616 | 61,71 % |

Alle drei beschleunigten Varianten bestehen die vorher festgelegten Grenzen. Beim neuen Startwert liegt der Prüfverlust um 0,12 % höher, Timing-F1 um 1,35 Prozentpunkte höher und der Bewegungsfehler um 0,10 % niedriger. Diese kleinen Unterschiede belegen weder eine Qualitätssteigerung noch statistische Gleichwertigkeit.

Auch die Einzelziele wurden angesehen: Geometrie-NLL verändert sich von −0,18446 auf −0,16691 (schlechter), der gewichtete Flow-Verlust von 0,00689 auf 0,00743 (schlechter), Slider-Dauer von 0,32925 auf 0,33044 (nahezu gleich). Die etwas bessere Rhythmusvorhersage gleicht diese Änderungen im Gesamtverlust aus. Daher keine Behauptung, dass jede Teilqualität unverändert bleibt.

Freie Generierung, je 128 Objekte desselben Prüfsongs: alle zwölf Läufe ohne zeitliche Überschneidungen. Referenz-/Graph-Mittelwerte: Timing-F1 55,87/56,27 %, mittlere Geometriekorrektur 0,00230/0,00296, stark korrigierte Objekte 0/0,26 %. Bei der Graph-Variante war damit eines von 384 Objekten stark zu korrigieren. Das 95%-Quantil der Sprünge liegt bei 143,8/175,3 Pixeln, der Bewegungsgeschwindigkeit bei 1.242/1.480 Pixel/s. Größere Sprünge sind nicht pauschal schlechter, zeigen aber: die erzeugten Bewegungsmuster sind nicht identisch. Ein einzelner Song und kurze untertrainierte Modelle reichen nicht, um gleiche Spielbarkeit zu bestätigen.

Numerisch: maximaler relativer Ausgabefehler Float32-SDPA gegenüber Referenz 0,000151 %, Gradienten-Kosinus 0,9999999999997. BF16: maximal 1,77 % relativer Ausgabefehler, Gradienten-Kosinus 0,9999803. Gemessen auf einem echten Batch ohne Dropout; keine Garantie für alle zukünftigen Batches.

Zusätzlich lief ein vollständiger CUDA-HTTP-Ablauf mit Graphs: zwei Trainingsepochen, gespeicherten Fused-Optimierer laden, dritte Epoche, Generierung und erneute Exportprüfung. 68 Objekte, davon 35 Slider, ohne Zeitüberschneidungen und außerhalb liegende Startpunkte. Alle gespeicherten Modellgewichte sind Float32. Dieser Dreisong-Lauf prüft ausschließlich die Technik. Zusätzlich: 60 automatisierte Tests und React-/TypeScript-Produktionsbuild erfolgreich; natives DirectML nach den Änderungen ebenfalls mit drei Epochen und Export erfolgreich geprüft.

## Nachstellen

Aus dem Repository, nach Einrichtung der lokalen CUDA-Umgebung und Import einer ausreichenden Library:

```powershell
$env:PYTHONPATH="$PWD/backend"
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TEMP="$PWD/data/tmp"
$env:TMP=$env:TEMP
& data/venv/Scripts/python.exe scripts/benchmarks/corpus.py
& data/venv/Scripts/python.exe scripts/benchmarks/cuda_benchmark.py parity
& data/venv/Scripts/python.exe scripts/benchmarks/cuda_benchmark.py quality --updates 120
& data/venv/Scripts/python.exe scripts/benchmarks/cuda_benchmark.py sustained --updates 150
& data/venv/Scripts/python.exe scripts/benchmarks/report.py
```

Manifest, Cache-Verweise, Umgebungsdaten, vorab definierte Grenzen, alle Einzelmessungen und Ergebnisbericht werden unter `data/benchmarks/cuda` gespeichert. Der Ordner ist von Git ausgeschlossen. Andere GPU-Aufgaben während der Durchsatzmessung vermeiden. Ein vorhandenes Manifest wird wiederverwendet; auf einem anderen Rechner den dortigen Library-Katalog vorbereiten.

Die Trainingsansicht misst später jede echte Epoche einschließlich Fortschrittsmeldungen und Checkpoints innerhalb der Schleife. Die erste Epoche enthält die Graph-Vorbereitung. Audio-Erstaufbereitung und Validierung werden separat ausgeführt; deshalb sind diese UI-Werte nicht direkt mit dem aufgewärmten Benchmark gleichzusetzen.

## Quellen

- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch Scaled Dot Product Attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch CUDA Graphs und Randbedingungen](https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs)
