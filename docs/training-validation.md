# Trainingssteuerung und Qualitätsprüfung – 27. Oktober 2026

## Änderungen

- Warmup unabhängig von der maximalen Epochenzahl; anschließend Plateau-Halbierung statt einer an 100 Epochen gebundenen Kosinuskurve. Optimierer-Checkpoint enthält Lernrate, Warmup- und Frühstopp-Zähler.
- Unveränderte zwölf Hauptlernziele und unveränderte Auswahl nach niedrigstem Gesamt-Prüfverlust. Separate Gruppenwerte und beste Gruppenepochen machen gegenläufige Entwicklungen sichtbar.
- Freie Validierung auf bis zu drei festen, unterschiedlichen Song-Gruppen in Epoche 1, alle fünf Epochen und am Ende. Je maximal 256 Objekte mit eigener Vorgeschichte, fester Seed. Separater Abschlusstest erst nach Laden der besten Gewichte. Fehlgeschlagene Generierungen werden mitgezählt und angezeigt.
- Optionaler zweiter Decoderlauf mit begrenzten eigenen Positionsvorschlägen: ab Epoche 5, maximal 20 % der Vorgänger, höchstens 16 Pixel Verschiebung, Zusatzgewicht 0,1. Hauptpfad und Referenzdaten bleiben unverändert; keine zusätzlichen direkten Audio-Encodergradienten. **Standardmäßig aus**, da der Vergleich keinen konsistenten Zusatzgewinn zeigt.

## Exploratorischer A/B/C-Vergleich

`scripts/benchmarks/training_policy.py` verwendet ausschließlich einen vorhandenen Cache: 32 echte Maps/Song-Gruppen, davon 24 Training, 5 Validierung, 3 Abschlusstest. RTX 5070 Ti, großes Modell (33,4 Mio. Parameter), BF16, CUDA Graphs, Batch 8, Seeds 42 und 73, je höchstens 24 Epochen. Ausgangsgewichte und Datenreihenfolge je Seed gleich initialisiert; CUDA-Ausführung ist nicht bitweise deterministisch. In jedem Lauf wird die Epoche nur anhand der Validierung gewählt. Die alte Kosinuskurve bleibt wie im Nutzerlauf auf 100 Epochen ausgelegt, der Vergleich wird aber nach 24 begrenzt. Der neue Warmup ist auf diesem kleinen Korpus kürzer; dies ist ein Vergleich der gesamten Trainingssteuerung, keine isolierte Plateau-Ablation.

| Verfahren | Seed | Beste Epoche | Test-Gesamtverlust ↓ | Test-Bewegungsfehler ↓ | Freie Zeitpunkte-F1 ↑ |
|---|---:|---:|---:|---:|---:|
| Bisherige Steuerung | 42 | 15 | 1,2773 | 0,2331 | 50,43 % |
| Plateau-Steuerung | 42 | 19 | 1,1367 | 0,2260 | 46,96 % |
| Plateau + Zusatztraining | 42 | 19 | 1,2017 | 0,2300 | 48,26 % |
| Bisherige Steuerung | 73 | 24 | 1,1712 | 0,2265 | 48,34 % |
| Plateau-Steuerung | 73 | 19 | 1,1482 | 0,2181 | 48,64 % |
| Plateau + Zusatztraining | 73 | 24 | 1,1231 | 0,2205 | 48,39 % |

Gesamt-Testverlust im Mittel: bisher 1,2242, Plateau 1,1424, Plateau + Zusatztraining 1,1624. **Die freie Zeitpunkte-F1 sinkt im Mittel von 49,38 % auf 47,80 % bzw. 48,32 %.** Das ist kein Nachweis einer durchgängigen Qualitätsverbesserung. Die neue Steuerung verbessert in dieser Stichprobe Gesamtverlust und Bewegungsfehler, aber die bisherige Gesamtscore-Auswahl bleibt nur ein Ersatzmaß für freie Generierungsqualität. Die Zusatzoption bleibt wegen des gemischten Ergebnisses aus. Alle 18 freien Prüfabschnitte ohne zeitliche Überschneidungen oder von den bestehenden Regeln erkannte Slider-Formprobleme; das ist wegen der gemeinsamen Generierungsregeln kein unabhängiger Qualitätsbeleg.

Medianer Trainingsdurchsatz nach der Anlaufphase: etwa 162–166 Audiofenster/s ohne Zusatzpfad, 136–138 mit Zusatzpfad. Maximal belegter Torch-GPU-Speicher etwa 1,25 GiB gegenüber 1,87 GiB. Werte ohne Validierung/freie Generierung und ohne Checkpoint-Dateischreiben dieses Benchmarks; zeitweise parallele technische Tests können Durchsatz beeinflussen. Rohdaten: `data/benchmarks/training-policy/report.json`.

Diese sechs kurzen Läufe dienen einer exploratorischen Regressionseinschätzung, keinem endgültigen Qualitätsnachweis. Nur drei Abschlusstestsongs, keine menschlichen Spieltests, keine statistisch belastbare Genre-Abdeckung, teils noch fallende Verluste bei Epoche 24. Die Benchmark-Ergebnisse wurden zur Entscheidung über den experimentellen Standard verwendet; ein unabhängiger Bestätigungssatz bleibt nötig. Keine Nutzergewichte überschrieben, keine neuen Modelle oder Trainingsjobs im produktiven Katalog registriert.

## Technische Prüfungen

68 gezielte automatisierte Tests erfolgreich (in drei fachlichen Gruppen ausgeführt, neue Tests nach der abschließenden Voreinstellungsänderung erneut geprüft). Produktionsbuild, Black-/Prettier-Prüfung und `git diff --check` erfolgreich. Vergleichsprozesse und isolierter UI-Testserver danach beendet.

- Lernrate vor Frühstopp reduziert; negative Verluste und fehlende Validierung korrekt behandelt; Steuerzustand nach Fortsetzen erhalten.
- Echte Trainings-/Fortsetzungsschleife einschließlich Aktivierung des Zusatzpfads ab Epoche 5, Gradient Accumulation und unveränderter bester Inferenzdatei.
- Kausalität, BOS/Padding, starre begrenzte Verschiebung, unveränderte Referenzdaten und Hauptausgaben sowie endliche Gradienten auf CPU und CUDA.
- Zusatzpfad mit dynamischer Wahrscheinlichkeit unter BF16/CUDA Graphs; nativer DirectML-Vorwärts-/Rückwärts-/Optimiererschritt erfolgreich (bekannter AdamW-CPU-Fallback für `lerp`).
- Feste unterschiedliche Validierungssongs, Fehlerzählung und Abbruchweitergabe; getrennte Testsongs nur im Abschluss.
- Bestehende CUDA-, Datenaufbereitungs-, Lernfähigkeits-, Modellimport-, Slider-, Checkpoint- und vollständige HTTP-Trainings-/Generierungstests.
- React/TypeScript-Produktionsbuild und Browserprüfung mit isolierten UI-Prüfdaten: Auswahl der Verlustgruppen, negative Geometriewerte, beste/letzte Epoche, Lernrate, freie Diagnostik und abschaltbare Zusatzoption.

## Einordnung

Überwachtes Lernen aus Audio und guten Referenzmaps, musikalischer Kontext und kausale Objektfolgen sind eine sinnvolle Grundlage. Die Änderungen verbessern Steuerbarkeit und Fehlerdiagnose. Sie belegen weder Spitzenqualität noch Gleichwertigkeit mit erfahrenen Mappern. Hauptgrenzen bleiben die zweistufige Zeit-/Objektplanung, Viersekunden-Dichtebudgets, eingeschränkte Sliderdarstellung und überwiegend referenzbasierte Vorgeschichte. Vor weiterem Architekturumbau sollten vollständige neue Songs mit gleichem Modell/Seed/Parametern gespielt und Rhythmus, Bewegungen sowie Reparaturhäufigkeit getrennt bewertet werden.

## Nachtrag 28. Oktober 2026: historischer v3-Vergleich und neuer v5-Stand

Die obenstehenden Änderungen, Messwerte und 68 Prüfungen beschreiben den
**historischen v3-Stand vom 27. Oktober**. Dessen A/B/C-Bezeichnung meint die drei
Varianten der Trainingssteuerung und ist nicht mit dem neuen Architekturvergleich
A (bisherige Inferenz), B (neue Inferenz mit denselben v3-Gewichten) und C (neue
Repräsentation mit neu trainierten Gewichten) gleichzusetzen. Die alten Zahlen
bleiben als Messergebnis erhalten; sie validieren keine neuen v5-Komponenten.

Der aktuelle [Architekturstand](architecture.md) ist `phrase-mapper-v5`, Format 5,
Targetschema `musical-plan-targets-v3` und Cachemanifest 3. Er verwendet explizite
musikalische Querymerkmale mit Unsicherheitsmasken, einen kleinen Planhead auf
vorhandenen Songtokens, zusammenhängende Rhythmusauswahl und begrenzte gemeinsame
Zeit-/Sliderplanung. Feste Viersekundenbudgets der Inferenz entfallen; das
Viersekunden-Dichtetarget bleibt vorerst als vorhandene Trainingshüllkurve.
Freie Diagnostik prüft feste spätere Abschnitte und begrenzte Vollsongs anstelle
reiner Präfixe. [Entscheidungen und offene Punkte](architecture-roadmap.md).

Der neue Kontextbewerter erhält strukturgleiche, geometrisch plausible
Kontrafaktuale mit schwachen Präferenzen. Das weiterhin ausgeschaltete
Eigenkontexttraining lernt unter veränderter Historie nur Typ, Dauer, Repeats und
Combo; die damalige zusätzliche absolute Geometriesupervision entfällt. Plan-
und Kontextverluste bleiben außerhalb der Checkpoint-Auswahlmetrik der zwölf
Hauptziele. Technische Tests dieser Änderungen sind im
[Zeit-/Planbericht](musical-time.md) und im Umsetzungsregister dokumentiert.

Der neue C-Vergleich über `scripts/benchmarks/architecture_training.py` ist mit
zehn Läufen (fünf Varianten, zwei Seeds) abgeschlossen. Er trennt Sekundenbasis, musikalische
Zeit, Plan, Kontextbewerter und Eigenkontexttraining mit gleichen Songgruppen,
Parameterkapazitäten, Updatebudgets und mehreren Seeds. Ausgangspunkt sind neue
Gewichte; es gibt keine v3-Konverter. Ein Code-Snapshot und Dateihashes sichern
den Versuchsstand. Der bisherige v3-Gewichtsstand hat ein anderes Trainingsbudget
und ist deshalb kein fairer Qualitätsranglistenvergleich mit den kurzen C-Läufen.

Vollständige neue Ergebnisse und Einschränkungen werden ausschließlich im
[Architektur-Evaluationsbericht](architecture-evaluation.md) zusammengeführt.
Der vorhandene Korpus enthält keinen frischen unexponierten Abschlusstestsatz;
historische Testgruppen werden nicht umetikettiert. Menschliche Bewertungen
bleiben bis zu tatsächlich durchgeführten verblindeten Playtests offen.
