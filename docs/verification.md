# Lokaler Prüfstand

Stand: 26.10.2026. Geprüfte Architektur: **Phrase Mapper v3**. Diese Prüfungen bestätigen Funktionsfähigkeit und technische Invarianten, noch keine gute spielerische Qualität.

## Tatsächlich ausgeführte GPU-Läufe

Jeweils drei Epochen mit dem vollständigen Standardmodell (11.316.079 Parameter), drei vorhandenen Maps, zwei Trainingssongs und einem Validierungssong. Anschließend freie Testgenerierung und Generierung aus einer separaten hochgeladenen Audiodatei mit `.osz`-Export.

| Messung | CUDA | Natives DirectML |
| --- | ---: | ---: |
| Trainingsverlust, erste → dritte Epoche | 4,032 → 3,056 | 4,079 → 3,023 |
| Prüfverlust, erste → dritte Epoche | 3,364 → 3,061 | 3,226 → 3,043 |
| Roher Onset-F1, dritte Epoche | 0,226 | 0,216 |
| Bewegungsfehler, dritte Epoche | 0,293 | 0,302 |
| Exportierte Objekte / Slider | 89 / 49 | 88 / 50 |
| Freier Prüfabschnitt: Objekte | 59 | 57 |
| Freier Prüfabschnitt: Zeitpunkte-F1 | 0,259 | 0,211 |
| Freier Prüfabschnitt: stärker korrigierte Platzierungen | 35,6 % | 35,1 % |
| Zeitliche Überschneidungen | 0 | 0 |

Rohprotokolle: `data/verification-phrase-cuda.json` und `data/verification-phrase-directml.json`. Die Zahlen zeigen ausdrücklich **kein ausgereiftes Modell**: viel zu wenige Songs, viele nötige Korrekturen, schwaches Timing. Das lässt sich nicht durch die Bezeichnung „Transformer“ oder sinkenden Verlust wegdiskutieren. Der Nutzer muss ein größeres kuratiertes Modell trainieren und spielen; dafür ist die Pipeline vorbereitet.

Hardware: NVIDIA GeForce RTX 5070 Ti. CUDA verwendet die bereits installierte PyTorch-Version `2.8.0.dev20250314+cu128`; Setup/Container pinnen regulär 2.8.0. Natives DirectML verwendet PyTorch `2.4.1+cpu` mit torch-directml `0.2.5.dev240914` und tatsächlich `privateuseone:0`. Die `lerp`-Operation des Optimierers fällt bei diesem Anbieter auf CPU zurück. Der Vorwärts-/Rückwärtslauf des Netzes funktioniert; beim Architekturumbau gefundene Rückwärtsprobleme der Stabilisierung wurden beseitigt.

## Automatisierte Prüfung

60 Tests, isolierte Arbeitsverzeichnisse unter `data/tests` und `data/pytest-*`. Abgedeckt sind:

- CUDA: SDPA-Ausgaben/-Gradienten und Kausal-/Paddingmasken, Graph-Replay mit neuen Eingaben und geänderten Gewichten, ersetzte Gradienten, nicht eingefrorener Dropout und erhaltene Zufallszustände bei Vorbereitung. Prefetch-Reihenfolge/-Fehler, Nicht-Endlichkeitsabbruch, vektorisierte Zielparität und Fused-AdamW-Resume.
- Vollständiger HTTP-Ablauf: Archivimport → Datensatz → Training → Modell-Download/Wiederimport → Audio-Upload → Generierung → Map-Download → Auswahl exportieren/reimportieren.
- Kein Zugriff auf zukünftige Zielpositionen durch den kausalen Decoder. Änderungen älterer Vorgänger und ferner Audioabschnitte beeinflussen Ausgaben.
- Alle zwölf Lernziele: endliche Gradienten und messbarer Verlustabfall beim gezielten Anpassen eines kleinen Batches. Das ist ein Lernfähigkeitstest, kein Generalisierungsbeweis.
- Gemeinsame Mischverteilungen bevorzugen getrennte plausible Layouts gegenüber deren Mittelwert; stabile kategorielle/binäre Losses stimmen mit PyTorch-Werten und -Gradienten überein.
- Bewegungsberechnung in Training und Kandidatenauswahl stimmt überein; Prüfverlust ist unabhängig von der Batch-Aufteilung. Zeitpunkte-Matching zählt nahe Treffer ohne doppelte Zuordnung oder Wegnehmen des nächsten gültigen Treffers.
- Lange Objektfenster decken jedes Ziel genau einmal ab, Rhythmusverlust wird nicht durch mehrere Objektteile übergewichtet, Beginn-/Padding-/Spinner-Masken und gespiegelte Vorgeschichte bleiben konsistent.
- Song-Gruppen sind zwischen Training/Validierung/Test disjunkt. Änderungen einer Map trotz gleichbleibender Objektanzahl verhindern irrtümliches Fortsetzen.
- Gesicherter AdamW-Zustand erzeugt nach Wiederherstellung dasselbe nächste Update. Abbruch erhält ein fortsetzbares Modell; Neustart erkennt zurückgelassene Checkpoints genau einmal.
- Kubische Slider mit 1–4 Durchläufen, variable BPM/SV, Triolen-Dauern, tatsächliche Endpunkte und Tempoänderungen überstehen den Writer-/Parser-Roundtrip. Korrekturen berücksichtigen CS-Ränder, freie Zeit und Slider-Enden.
- Synthetische Beatvorhersagen erkennen Tempowechsel. Mehrminütiger Timingtest verhindert Drift durch Frame-Quantisierung. Digitale Stille erhält auch nach Rasterkorrektur keine Noten.
- Deklarierte Slider-Pfadlängen, positive Offsets, defekte Audio-Caches und ungeeignete Trainingspaare werden korrekt behandelt.
- ZIP-Slip, ungültige Archivpfade, fremde Origins/Hosts und nicht-endliche Modellgewichte werden abgewiesen. Alte Architekturen werden explizit abgelehnt.
- Abbruch/Fehler räumen unfertige Exporte/Datensätze auf; veröffentlichte Ergebnisse bleiben erhalten. Ohne geeignete Slider im Trainingssatz wird Circles-only generiert.

React-/TypeScript-Produktionsbuild erfolgreich. Die neue Trainingsansicht, Modellgröße, Fortsetzen-Auswahl, freie Modellberichte, Bewegungsparameter und Variantenoptionen wurden im Browser geprüft. Die Vorschau zeigt gekrümmte Slider inklusive Sliderball; die Wiedergabeposition lässt sich ändern. CUDA-Startprüfung mit Loss/Rückwärtslauf/Update erfolgreich. Beide GPU-Compose-Dateien syntaktisch validiert.

## CUDA-Optimierung und Messung

Zusätzlicher Vergleich auf der RTX 5070 Ti: 32 echte Maps, 24 Training / 5 Validierung / 3 unbenutzte Testsongs, zwölf Läufe über drei Seeds mit jeweils 120 Updates. Numerische Parität geprüft; alle vorab definierten Grenzen für Prüfverlust, Timing-F1 und Bewegungsfehler eingehalten. Einzelne Geometrie-/Flow-Teilwerte und frei generierte Muster unterscheiden sich weiterhin. Kein Langzeit- oder menschlicher Spielbarkeitstest.

Drei Durchsatzwiederholungen je Einstellung: Median 19,1 Fenster/s im Referenzpfad, 330,9 mit BF16/Graphs bei Batch 8. Reservierter PyTorch-Speicher maximal 0,99 GiB; gesamter beobachteter Gerätespeicher bis 3,17 GiB. Batch 32: 450,2 Fenster/s, ohne gesonderten Qualitätsnachweis. Details, Messgrenzen und alle Qualitätskennzahlen im [CUDA-Messbericht](cuda-performance.md); Rohdaten unter `data/benchmarks/cuda`.

Optimierter CUDA-Pfad zusätzlich über das echte HTTP-Backend geprüft: Training, Checkpoint, Fortsetzung und Export. DirectML nach den Änderungen erneut nativ trainiert und exportiert: drei Epochen, 88 Objekte mit 50 Slidern, keine Zeitüberschneidungen; Protokoll `data/verification-cuda-compat-directml.json`. Leistungsanzeige und CUDA-Standardwerte im Browser sichtbar geprüft. CUDA-Startprüfung erfolgreich. React-/TypeScript-Produktionsbuild und 60 Tests erfolgreich. Keine neuen Downloads.

## Library-Filter und Vorschau

Sternefilter und Gesamtauswahl nutzen dieselbe serverseitige Abfrage. 23 gezielte Tests erfolgreich: inklusive Grenzen, mehr als zwei Ergebnisseiten, einseitige Filter, unbekannte/ungültige Sterne, Pfadschutz in API-Antworten, Bewertungserhalt bei erneutem Import sowie bestehende Import-/Training-/Generierungs-, Beatmap- und Jobtests. React-/TypeScript-Produktionsbuild und erneuter C#-Reader-Build erfolgreich.

Alle 3.348 vorhandenen lazer-Katalogeinträge haben jetzt die in der lokalen Datenbank gespeicherte Sternebewertung. Im Browser geprüft: 4–7 Sterne ergeben 1.264 Treffer, Gesamtauswahl bleibt auf der nächsten Seite erhalten, Filterwechsel leert die Auswahl, widersprüchliche Grenzen werden angezeigt und wiederholtes Zurücksetzen blockiert das Laden nicht. Größere Schrift, Sterne-Spalte und entfernter Header-Avatar sichtbar geprüft. Die Vorschau-Lautstärke lässt sich ändern und bleibt nach Neuladen/Ansichtswechsel erhalten.

Der gewünschte Reset wurde durch die automatische Freigabeprüfung blockiert; es wurden keine Dateien oder Datensätze gelöscht. `scripts/reset-training-data.ps1` ist für die manuelle Ausführung vorbereitet und syntaktisch geprüft, aber nicht ausgeführt. Es entfernt die bekannten Modelle, Trainingsdatensätze einschließlich kopierter Songs/Maps, Trainingsläufe und eigene Testartefakte. Originale in osu!lazer und der Map-Katalog bleiben erhalten, ebenso die fünf vom Nutzer erzeugten Beatmaps und zwei hochgeladenen Audios. Installierte Laufzeiten und Paket-Caches bleiben verfügbar.

## Einzelnes Löschen in der WebUI

Modelle und generierte Beatmaps besitzen einen Papierkorb mit nativem modalen Bestätigungsdialog, Fokus auf Abbrechen und Escape-Unterstützung. Die Kreationenliste ist mit zehn Einträgen pro Seite vollständig durchblätterbar. Die Auswahl fällt beim Entfernen einer Map auf eine vorhandene Vorschau zurück; veraltete Statusantworten vor dem Löschen können entfernte Einträge nicht erneut einblenden.

DELETE-Endpunkte entfernen ausschließlich eigene ID-basierte Modell-/Ergebnisdateien und zugehörige Datenbank-/Verlaufseinträge. Freie Dateipfade aus Metadaten werden dafür nicht verwendet. Modell-Checkpoints einschließlich Recovery-Dateien werden mit entfernt. Neue Trainingsstände speichern ihre Job-ID, damit auch unterbrochene/abgebrochene Läufe eindeutig zugeordnet werden können; ältere Läufe werden über vorhandene Ergebnis-IDs zugeordnet. Gemeinsame Datensätze, Uploads und andere Ergebnisse bleiben erhalten. Während aktiver oder wartender Jobs und ihrer abschließenden Bereinigung ist Löschen gesperrt. Aufnahme neuer Jobs, Prüfung verwendeter Modelle und Löschen teilen eine Sperre für den einzelnen Backend-Prozess.

Gezielte Tests prüfen echte Dateientfernung und Datenbankbereinigung, erhaltene gemeinsame Quellen/andere Artefakte, fehlende Dateien, einzelne Archivexporte, ungültige IDs, verknüpfte Unterordner, Dateisperren mit erneutem Versuch, aktive/abgebrochene/noch bereinigende Jobs und gleichzeitige Generierungsanfragen. Dazu bestehende Job-, Checkpoint-, Modellimport- und vollständige Trainings-/Generierungs-Pipeline-Prüfungen. React-/TypeScript-Produktionsbuild erfolgreich. Im Browser mit isolierten Testdaten geprüft: beide Dialogtexte, Abbrechen/Escape, Fokus-Rückgabe und zweite Seite mit älteren Maps. Keine Nutzermodelle oder Nutzer-Beatmaps wurden zum Test gelöscht.

## Slider-Kurvenkorrektur

Die Prüfung der vorhandenen Nutzermodelle zeigte sowohl eine sehr kleine Datengrundlage (17 Maps, elf Trainingssongs, je zwei Validierungs-/Testsongs) als auch ein unabhängiges Decoderproblem: acht voneinander unabhängig gezogene Koordinaten und punktweises Begrenzen an Spielfeldrändern konnten gelernte Slider-Kurven stark verformen. Die neuen Kandidaten erhalten eine vollständige gelernte Kurve unter gemeinsamen Transformationen. Eine zusätzliche konservative Formprüfung verwirft Selbstkreuzungen, scharfe Haken, extreme Umwege und Kontrollpunkt-Duplikate mit abweichender `.osu`-Segmentierungssemantik.

30 gezielte Tests bestanden: erhaltene Kurvenform trotz hoher Unsicherheit, unveränderte normale Bögen/S-Kurven, Randanpassung ohne Verbiegen, problematische Schleifen/Haken/kurze Umwege, sicherer Circle-Fallback, Slider-Wiederholungen und Tempoänderungen beim Export, bestehende Modell-Lernfähigkeit und komplette HTTP-Trainings-/Generierungspipeline. Ein isolierter Samplingvergleich mit 500 Varianten derselben sauberen Ausgangskurve ergab 105 auffällige Formen beim bisherigen unabhängigen Rauschen und null bei kohärenter Variation.

Zusätzlich sechs CUDA-Probegenerierungen auf zwei Nutzer-Audios mit den unveränderten Gewichten BobV1.1/BobV1 und Seeds 42/7/123, jeweils die ersten 128 Objekte: insgesamt 768 Objekte, davon 622 Slider. Keine von der neuen Prüfung erkannten Formprobleme und keine zeitlichen Überschneidungen. In den zuvor gespeicherten beiden 128-Objekt-Abschnitten waren 34 von 111 bzw. 23 von 110 Slidern auffällig. Die Onset-Zeitpunkte stimmen mit den bisherigen Ausgaben überein; Schreiben/erneutes Parsen bestätigt Dauern innerhalb 2 ms und Endpunkte innerhalb 1 Pixel. Rohdaten und Beispielkurven liegen unter `data/tests/slider-regression` und sind ausschließlich Testartefakte, keine neuen Modelle oder UI-Trainingsläufe.

Die Formkennzahlen verwenden dieselbe konservative Prüfung wie die Kandidatenauswahl und sind deshalb kein unabhängiger Spielbarkeitsnachweis. Stichproben der Kurven wurden zusätzlich visuell verglichen. Kein Training geändert, kein neuer allgemeiner Qualitäts- oder menschlicher Spieltest. Schwache Modelle können weiterhin stilistisch wenig überzeugende Kurven und Muster erzeugen; absichtlich verschlungene Spezialslider werden durch die Regeln eingeschränkt. Weitere Details in der [Architektur](architecture.md#generierung-und-spielbarkeit).

## Bewährte Importgrundlage

Der unabhängige C#-Reader wurde zuvor mit .NET SDK 10.0.301 und Realm 20.1.0 gebaut. Eine Kopie der echten lazer-Datenbank wurde gelesen: 3.348 osu!standard-Maps registriert, 366 Maps anderer Modi übersprungen. 27 problematische Einträge; untersuchte Dateien enthielten Nullbytes statt gültigem Text. Quelldateien wurden nicht verändert. Details im [Exporter-Audit](exporter-audit.md).

## Ergänzung: Song-Auswahl und Reglerhilfe (27. Oktober 2026)

14 Katalogtests bestanden: inklusive Sternegrenzen, Suche vor Schwierigkeitsauswahl, leichteste/mittlere/schwerste Map pro Song über mehrere Seiten, deterministische Gleichstände, fehlende Bewertungen und Audio-Hashes, transitive Songgruppen sowie ungültige Filter. Beide API-Routen verwenden dieselbe Auswahl. Frontend-Produktionsbuild und Formatprüfungen erfolgreich.

Browserprüfung mit isolierten Testdaten (107 Songs mit jeweils drei Schwierigkeiten): „Schwerste“ ergibt 107 Maps; Gesamtauswahl bleibt auf Seite zwei erhalten. Der Wechsel auf 4–7 Sterne leert die Auswahl und zeigt pro Song die passende 5-Sterne-Map. Neues Filterlayout visuell geprüft. Alle neun Generierungsregler haben Hilfe-Icons; Hilfetext beim Anklicken und Schließen mit Escape einschließlich Feineinstellungen geprüft. Screenshot: `data/tests/library-controls-ui/library-filters.png`. Keine Nutzerdaten, Modellgewichte oder Trainings-/Generierungsberechnungen verändert.

## Speicherverwaltung und Bereinigung (27. Oktober 2026)

40 unterschiedliche gezielte Tests bestanden: Speicherinventar, vollständige Artefaktlöschung, Dataset-/Upload-Löschung, Cache-Limit einschließlich deaktivierter Persistenz, Verlaufsbegrenzung, fehlgeschlagene Bereinigung mit erneutem Löschversuch, Pfad-/Symlinkschutz, Sperre bei aktiven Aufgaben und tatsächlichen Uploads, Import-Rollback, doppelte Importe und abgebrochene Exporte. Enthalten sind die bisherigen 14 Katalogtests und vier Pipeline-Tests einschließlich echtem Training/Generieren/Download. Nach letzten Backend-Anpassungen wurden die 22 Speicher-/Löschtests erneut erfolgreich ausgeführt. Frontend-Produktionsbuild erfolgreich. Browserprüfung mit isoliertem Datenverzeichnis: Kategorien und Größen, Löschdialog mit konkretem Umfang, Abbrechen erhält den Eintrag. Screenshot: `data/tests/storage-ui/storage.png`.

Produktivdaten auf explizite Nutzeranweisung bereinigt: drei Datensätze (740, 18 und 3 Maps), zugehörige Datensatz-Verlaufseinträge und gesamter Cache entfernt; 5,143 GiB freigegeben. Danach null Datensatz-Einträge und keine Dateien in `data/datasets` oder `data/cache`. Acht Modelle und fünf Ausgaben unverändert vorhanden. Ältere Test-/Benchmark-Dateien bleiben über die neue Speicherverwaltung gezielt entfernbar. Keine Trainingsziele, Modellgewichte oder Generierungsregeln verändert.

Die anschließende Entfernung der bei dieser Prüfung neu angelegten Testordner wurde durch die automatische Freigabeprüfung blockiert (keine genauere Begründung). Diese Testreste einschließlich isolierter UI-Testdaten bleiben unter `data/tests` und sind über die Speicherverwaltung entfernbar. Der UI-Testserver wurde beendet. Die separate Nutzerbereinigung von `data/cache` und `data/datasets` war erfolgreich und wurde danach nochmals bestätigt.

## Offene Qualitätsnachweise und Plattformgrenzen

Die neue Plateau-Steuerung, getrennte Qualitätsauswertung und optionale Positionsrobustheit wurden zusätzlich technisch und mit sechs kurzen CUDA-Vergleichsläufen geprüft. Ergebnisse einschließlich gemischter freier Generierungsqualität und experimentell ausgeschalteter Zusatzoption: [Trainingsvergleich vom 27. Oktober](training-validation.md). Kein pauschaler Nichtverschlechterungsnachweis.

- Noch kein großes Trainingskorpus durchtrainiert, kein belastbarer Vergleich gegen menschliche Mapper/andere Generatoren, keine ablationsbasierte Optimierung der zwölf Verlustgewichte.
- Keine bestätigte Spiel-/Importprüfung in der osu!lazer-Spieloberfläche. Programmatische Exportprüfung und Webvorschau ersetzen das nicht.
- Noch kein vollständiger CUDA-/DirectML-Docker-Image-Build: Docker-Speicher lag auf C:, während Downloads im Repository bleiben sollen. Compose und native Laufzeiten wurden geprüft.
- DirectML im WSL2-Container ist experimentell; GPU-Passthrough wurde nicht bestätigt. Natives DirectML wurde tatsächlich trainiert und getestet.
- Keine neuen Paket- oder Modellgewichtsdownloads bei diesem Architekturumbau. Testmodelle, Checkpoints und weitere Prüfartefakte liegen im ignorierten `data`-Ordner; der gewünschte Reset ist vorbereitet, aber nicht ausgeführt.
