# Architektur: Phrase Mapper

Stand: 26.10.2026. `phrase-mapper-v3`, Format 3, Ereignisschema `cubic-phrase-beats-v1`. Eine bewusst neue Architektur ohne Kompatibilität zu früheren Versuchsversionen. Alle trainierten Komponenten gehören zu **einem** Safetensors-Gewichtsstand; keine fremden vortrainierten Modelle erforderlich.

## Musikalischer Kontext

FFmpeg dekodiert zu 22.050 Hz Mono. 64 Log-Mel-Bänder, FFT 2048 und Hop 256 liefern etwa 11,61 ms je Frame. Ein absoluter RMS-Boden erhält digitale Stille trotz relativer Lautstärkennormalisierung. Dateiinhalte bestimmen die Cache-Schlüssel.

Pro Beispiel: 2.048 Frames (23,78 Sekunden), davon 1.024 zentrale Frames als Rhythmusziele. Zwei gelernte Faltungen verdichten auf 512 Audio-Tokens. Für den ganzen Song werden zusätzlich 128 zeitlich geordnete Abschnitte aus Mel-Mittelwert und -Maximum gebildet. Ein eigener Transformer verarbeitet diesen groben Überblick; die lokalen Audio-Tokens können anschließend auf diese globale Erinnerung zugreifen. Sinus-/Kosinus-Zeitmerkmale mit Perioden von 1/8 bis 16 Sekunden und relative Songposition geben zeitliche Orientierung. Sie sind keine fest vorgeschriebene Beat-Position.

Standard: Breite 256, acht Attention-Heads, zwei globale, sechs gemeinsame Audio- und vier Objektdecoder-Blöcke; 11.316.079 Parameter. Kompakt: 128 / 4 Heads / 4 Audio- / 3 Decoder-Blöcke, 2.283.631 Parameter. Groß: 384 / 8 Heads / 8 Audio- / 6 Decoder-Blöcke, 33.441.775 Parameter. Die Modellgröße ist keine Qualitätsrangliste; größere Modelle brauchen mehr geeignete Daten und Rechenzeit.

Ein hochauflösender Rhythmuszweig kombiniert hochgerechnete Transformer-Merkmale mit lokalen Audiodetails. Er sagt Onset-Wahrscheinlichkeit, Beat-Puls, örtliche Notendichte und logarithmische Beatlänge voraus. AR, CS, OD, HP, Dichte, typische Bewegungsweite, Richtungskontinuität und Slider-Anteil konditionieren den gemeinsamen Encoder. Stilwerte werden beim Training aus der jeweiligen Map gemessen.

## Zusammenhängende Hitobject-Folgen

Der kausale Decoder nutzt bis zu 128 Ereignisse. Jede Eingabe enthält ausschließlich das vorherige Objekt: Kurve, tatsächliches Endziel nach Wiederholungen, Dauer, Typ, Wiederholungen, Zeitabstand, Combo und Sprungweite. Die aktuelle Anfrage enthält den geplanten Notenzeitpunkt, vorige/nächste Zeitlücke und deren Verhältnis. Die nächste Position ist niemals Teil der Eingabe. Cross-Attention verbindet diese Vorgeschichte mit dem Audio. Im Training werden höchstens 64 neue Ziele pro Beispiel bewertet, mit älteren Objekten als Kontext. Rhythmusziele werden bei mehreren Objektteilen desselben Audiofensters nicht mehrfach gewichtet.

Gemeinsame Mischverteilungen mit acht Komponenten beschreiben Startposition und drei weitere Punkte einer kubischen Bézier-Kurve. Jede Komponente besitzt acht Mittelwerte und Skalierungen sowie ein Gewicht. Eine gemeinsame Komponentenauswahl erhält zusammengehörige Kurvenpunkte; mehrere Komponenten können alternative Platzierungen darstellen. Das vermeidet eine erzwungene Mittelung aller gültigen Muster auf eine einzige x/y-Position, verhindert aber keine Modenkollapse bei schlechtem Training.

Weitere Ausgänge: Circle/Slider, eine von 19 Beat-Dauern (inklusive Triolen, 1/12 bis 16 Beats), 1–4 Durchläufe, New Combo und acht Bewegungsgrößen. Diese Größen beschreiben Sprungweite, freie Bewegungsgeschwindigkeit, Winkel über Sinus/Kosinus, Geschwindigkeitsänderung, Slider-Reiseweg und -Geschwindigkeit sowie das Verhältnis aufeinanderfolgender Sprungweiten. Vertikale/horizontale Spiegelung erhält passende Bewegungsziele; leichte Störungen der vorherigen Position reduzieren Empfindlichkeit gegen kleine Generierungsfehler. Das bleibt überwiegend Teacher Forcing, kein Training vollständig auf eigenen Rollouts.

## Zwölf Lernziele

| Ziel | Bewertung | Gewicht |
| --- | --- | --- |
| Onsets | Gewichtete binäre Kreuzentropie; benachbarte Frames als weiche Ziele | 1 |
| Beat-Puls | Binäre Kreuzentropie auf gültigen Timingabschnitten | 0,3 |
| Lokale Dichte | Robuster Fehler der Dichtekurve über vier Sekunden | 0,5 |
| Tempo | Robuster Fehler der logarithmischen Beatlänge | 0,2 |
| Gemeinsame Geometrie | Maskierte negative Log-Likelihood der Mischverteilung | 1 |
| Objekttyp | Kategoriale Kreuzentropie | 0,5 |
| Slider-Dauer | Kategoriale Kreuzentropie, unterstützte Slider | 0,3 |
| Wiederholungen | Kategoriale Kreuzentropie, unterstützte Slider | 0,2 |
| New Combo | Binäre Kreuzentropie | 0,1 |
| Bewegung | Robuster Fehler der acht vorhergesagten Bewegungswerte | 0,3 |
| Bewegungsfolgen | Aus Kandidatengeometrie berechnete Bewegungen gegen Referenz | 0,15 |
| Kontextverträglichkeit | Richtige gegenüber vertauschter Geometrie bei gleichem Kontext | 0,1 |

Die letzten beiden Ziele bestrafen Abweichungen vom jeweiligen Trainingsmuster, nicht pauschal schnelle Bewegungen oder Richtungswechsel. Der Kontextbewerter lernt relative Zuordnung, keine menschlichen Bewertungen. Verlustgewichte und Grenzwerte sind Entwicklungsannahmen, nicht durch umfangreiche Qualitätsexperimente optimiert. Die Geometrie-NLL kann negativ sein; das ist bei kontinuierlichen Dichten zulässig.

Training: AdamW, Weight Decay, Gradient Accumulation, Gradient-Clipping auf Norm 1 und ein von der Epochenobergrenze unabhängiger Warmup (höchstens 200 Updates, höchstens eine Epoche). Danach halbiert die Steuerung bei drei Validierungen ohne Verbesserung um mindestens 0,0001 die Lernrate, bis auf 5 % der Ausgangsrate. Bei kürzerer Frühstopp-Geduld verkürzt sich das Intervall entsprechend. Jede echte Verbesserung sichert weiterhin das beste Modell; der Frühstopp zählt zwölf Epochen ohne neuen Bestwert und gewährt nach einer Reduktion mindestens ein Reduktionsintervall. Lernrate, Zähler und Warmup werden mit dem Optimierer atomar gespeichert. Fortsetzen bewahrt sie; eine ausdrücklich geänderte Ausgangslernrate skaliert die gespeicherte Rate proportional. Alte Checkpoints beginnen mit höchstens ihrer zuletzt protokollierten Lernrate. Ohne Validierung gibt es weder Plateau-Reduktion noch Frühstopp.

Nicht-endliche Verluste, Gradienten und Gewichte stoppen Updates. Numerisch stabile Grundoperationen umgehen bekannte DirectML-Rückwärtsprobleme; `max` zur Stabilisierung wird vom Gradientengraphen getrennt. DirectML kann die Optimiereroperation `lerp` auf CPU ausführen. Keine zwingende CUDA-spezifische Attention-Erweiterung.

### Zusatztraining für eigene Positionsvorhersagen

Optionales, **standardmäßig ausgeschaltetes** Zusatztraining ab Epoche 5: Die Auswahlwahrscheinlichkeit für geeignete Vorgänger steigt über zehn Epochen von 2 auf höchstens 20 %. Eine abgetrennte Vorhersage des **vorherigen** Objekts verschiebt dessen Start, Kontrollpunkte und wiederholungsabhängigen Endpunkt gemeinsam, maximal 16 Pixel. BOS/Padding bleiben erhalten, der Abstandsindikator wird nachgeführt. Zeiten, Typ und Dauer der Vorgeschichte bleiben referenzbasiert. Es handelt sich ausdrücklich um begrenzte modellbasierte Positionsstörungen, nicht um vollständig frei ausgerollte Trainingsfolgen. Der kleine A/B/C-Vergleich zeigte keinen konsistenten Zusatznutzen gegenüber der neuen Lernratensteuerung allein; daher bleibt die Option experimentell.

Der normale überwachte Vorwärtslauf und seine zwölf Ziele bleiben erhalten. Ein zusätzlicher Decoderlauf lernt Position/Kurve, Typ, Dauer, Wiederholungen und Combo mit Gewicht 0,1 unter der veränderten Vorgeschichte. Der Audiokontext wird dafür vom Gradientengraphen abgetrennt; es gibt keinen zusätzlichen direkten Encodergradienten. Bewegungs- und Kritikerziele bleiben auf der unveränderten Referenzvorgeschichte, um widersprüchliche Ziele zu vermeiden. Indirekte Veränderungen späterer Optimierungsschritte sind dennoch möglich. Der Zusatzverlust ist separat protokolliert und nicht Teil der vergleichbaren Hauptkurven oder Checkpoint-Auswahl. Der CUDA-Graph wird beim Aktivieren einmal neu aufgezeichnet und liest danach die steigende Wahrscheinlichkeit als Tensor. Abschaltbar für kontrollierte Vergleiche.

## Stil-Conditioning v2: Sterne, Patterns, Mapper

Neue Modelle (`style_version: 2` in `model_config`) ersetzen die acht festen Stilwerte durch 26 gemessene Größen mit Maske. Alle Werte stammen aus der jeweiligen Trainingsmap selbst (`backend/app/ml/style.py`):

| Gruppe | Werte |
| --- | --- |
| Schwierigkeit | offizielles Star Rating, Aim-Anteil an Aim + Speed |
| Einstellungen | AR, CS, OD, HP |
| Dichte/Tempo | Objekte pro Sekunde, dominantes BPM |
| Rhythmus | Anteil 1/4-, 1/2- und Triolen-Abstände, Bursts (3–8 Noten in 1/4-Folge), Streams (ab 9), mittlere Folgenlänge |
| Bewegung | Median-Abstand, 1/2-Sprungweite, Stream-Abstand, Richtungsfluss, spitze Winkel (< 60°), gerade Linien (> 120°), Stacks |
| Slider | Anteil, Tempo in px/Beat, kurze Slider (≤ 1/2 Beat), Wiederholungen |
| Struktur | Objekte pro New Combo |

Eingang des Encoders sind maskierte Werte plus Maske (52 Zahlen). Beim Training werden ganze Gruppen zufällig verborgen (Schwierigkeit 15 %, Rhythmus/Bewegung/Slider je 35 %, Struktur 50 %, Tempo 10 %, AR/CS/OD/HP nie). Dadurch funktioniert die Generierung auch, wenn nur Ziel-Sterne oder nur ein Mapper vorgegeben sind; fehlende Werte leitet das Modell aus Musik und Restvorgaben ab.

**Star Rating.** `difficulty/` ist ein kleines .NET-Programm mit `ppy.osu.Game.Rulesets.Osu` (2026.1026.0, inklusive Reading-Skill und Snap/Flow-Aim aus dem Rework vom Juli 2026). Es berechnet No-Mod-Attribute für viele `.osu`-Dateien parallel und schreibt JSON-Zeilen. Import, Datensatz-Vorbereitung und Generierung nutzen dieselbe Berechnung. Ohne gebautes Programm trainiert das Modell mit maskierten Sternen und warnt.

**Mapper.** Der Creator eines stable-Sets ist auch bei Gast-Difficulties eingetragen. Possessive Diff-Namen („NiNo's Insane“, „deetz' Expert“) werden deshalb dem Gast zugeordnet, bekannte Schreibweisen vereinheitlicht. Mapper mit mindestens `mapper_min_maps` Trainingsmaps (Standard 10) erhalten einen Eintrag in einer gelernten Tabelle (One-Hot-Produkt statt Embedding-Lookup, damit DirectML funktioniert); seltene Mapper und 30 % der Beispiele nutzen den neutralen Eintrag 0. Pro Mapper und pro halbem Stern speichert das Modell zusätzlich Median-Profile, mit denen die Oberfläche Regler vorbelegt.

**Rhythmus-Patterns.** Jede Map liefert pro Objekt die Rhythmusklasse des Abstands zum nächsten Objekt: 26 Beat-Abstände von 1/8 bis 8 sowie „Pause“ (über 8,5 Beats). Der Decoder sieht in der Anfrage nur noch den Abstand zum vorigen Objekt (Sekunden und Beats) und das Tempo, nicht mehr den zum nächsten. Ein eigener Kopf sagt aus Vorgeschichte, Audio und dem gerade platzierten Objekt (Circle oder Slider samt Länge) die nächste Rhythmusklasse voraus; Verlust mit Label Smoothing 0,05 und Gewicht 0,5, als eigene Kurve „Rhythmus-Patterns“. Beim Generieren wird der nächste Zeitpunkt so gewählt: Klassenwahrscheinlichkeit plus 0,7 × log Onset-Wahrscheinlichkeit am Rasterpunkt, nur Klassen auf dem gewählten Raster (1/12 mischt 1/4 und 1/3), nicht in digitaler Stille, Slider-Ende plus ¼ Beat Abstand, optional maximale Streamlänge. Damit entstehen Triples, Bursts, Streams und 1/2-Abschnitte als zusammenhängende Folgen statt als unabhängig ausgewählte Onsets.

**Gegen Overfitting.** Der Onset-Zweig bekommt in v2 Dropout (0,1) auf Transformer- und lokale Merkmale, das Mel-Spektrum bis zu zwei maskierte Frequenzbänder. Freie Validierung nutzt das Timing der Referenzmap, damit sie Patterns statt Tempoerkennung misst.

**Kalibrierung.** Mit Ziel-Sternen misst die Generierung die fertige Map. Liegt sie mehr als 0,15 ★ daneben, werden Sterne-Vorgabe und eine Rhythmus-Tendenz nachgeregelt und neu erzeugt (bis zu fünf Runden, beste Runde gewinnt). Ältere Modelle regeln stattdessen Dichte und Abstand nach.

**Startpunkt.** Ein neues Training kann Gewichte eines vorhandenen Modells übernehmen, auch eines älteren v1-Modells. Übernommen wird jeder Tensor mit passender Form; Conditioning, Mapper-Tabelle und Rhythmus-Kopf starten neu. Fortsetzen (Resume) ist nur für v2-Stände möglich.

## CUDA-Ausführung

CUDA verwendet PyTorch SDPA mit denselben Padding-/Kausalmasken; CPU und DirectML behalten die explizite Attention. Automatische Rechengenauigkeit nutzt auf geeigneter CUDA-Hardware BF16 nur im Netzvorwärtslauf. Die Ausgänge werden vor den zwölf Verlusten nach Float32 zurückgewandelt; insbesondere Mischverteilungen, Bewegungsableitungen und Kontextbewertung bleiben Float32. Mastergewichte, AdamW-Zustand, Validierung und Modell-Dateien bleiben Float32. TF32 ist deaktiviert. „Float32“ in der UI ermöglicht einen Vergleich ohne BF16.

CUDA-AdamW nutzt den integrierten Fused-Pfad. Ein Hintergrundthread bereitet maximal einen Folgebatch in gepinntem RAM vor; Transfers erfolgen nicht blockierend. Bewegungsziele werden vektorisiert berechnet. Verluststatistiken sammeln sich abgetrennt vom Gradientengraphen auf der GPU, statt jeden Teilverlust einzeln zur CPU zu übertragen. Endlichkeitsprüfung und Gradient-Clipping bleiben vor jedem Update erhalten; Fortschrittsmeldungen werden auf ungefähr einmal pro Sekunde begrenzt.

Bei aktivierten CUDA Graphs und genau einem Batch pro Update werden Vorwärtslauf, Verluste und Rückwärtslauf einmal aufgezeichnet. Wiederholungen lesen neue Daten und aktuelle Gewichte, verwenden neue Dropout-Masken und überschreiben die Gradienten. Optimierer, Lernrate, Clipping, Abbruch und Checkpoints bleiben außerhalb des Graphs. Abweichend große letzte Batches laufen normal; Gradient Accumulation verwendet ebenfalls den normalen CUDA-Pfad. Der Graph hält feste Puffer und einen privaten Speicherpool, braucht daher mehr reservierten Speicher als ein Lauf ohne Graph. Vorbereitungsdurchläufe ändern keine Gewichte; die Zufallszustände werden danach wiederhergestellt.

Pro Epoche: gemessene Trainingsdauer, Beispiele/s, Updates, Genauigkeit und CUDA-Spitzenwerte für belegten/reservierten Speicher. Erste Epoche inklusive Graph-Vorbereitung, ohne vorherige Audioaufbereitung und nachfolgende Validierung; innerhalb der Schleife geschriebene Checkpoints zählen mit. Das sind keine GPU-Gesamtspeicherwerte: Treiber, Desktop und andere Programme kommen hinzu. [Benchmark und Grenzen des Qualitätsnachweises](cuda-performance.md).

## Datengüte und unabhängige Prüfung

Audio und Map-Inhalt werden gehasht. Audio-SHA und normalisierte Künstler-/Titelangaben bilden transitive Song-Gruppen; diese bleiben vollständig im Training, der Validierung (~15 %) oder dem Test (~10 %, ab fünf Gruppen). Kein akustischer Fingerprint: anders codierte Dubletten mit abweichenden Metadaten können unerkannt bleiben. Der Split hängt am Seed. Validierung wählt den besten Gewichtsstand; der zusätzliche Test beeinflusst die Auswahl nicht.

Parser berücksichtigen Timingpoints, SliderMultiplier, vererbte Geschwindigkeit, tatsächliche Pfadlänge und Wiederholungsparität. Pfade werden auf einen kubischen Bézier approximiert. Maximal acht Pixel Anpassungsfehler, höchstens vier Durchläufe und 16 Beats sind die unterstützte Geometrie. Schwer darstellbare Slider liefern weiterhin Startposition und Typ, aber keine falschen Kurven-/Dauerziele. Spinner-Zeiträume werden maskiert. Ungültige Koordinaten, Treffer außerhalb des Audios und Kollisionen unter der Zeitauflösung schließen Maps mit begründeter Diagnose aus.

Die Validierung aggregiert nach gültigen Frames/Objekten. Sie meldet Teilverluste, rohe Onset-Präzision/Recall/F1 mit Eins-zu-eins-Matching ±35 ms, Objekttypgenauigkeit und Bewegungsfehler unter richtiger Vorgeschichte. Die UI zeigt getrennte Kurven für Rhythmus, Geometrie, Objektstruktur und Bewegung/Zusammenhang sowie Gesamtverlust, Lernratenwechsel, gespeicherte beste Epoche und tatsächliche Song-Aufteilung. Beste Gruppenwerte werden zusätzlich in den Modellmetadaten protokolliert. Die Auswahl bleibt bewusst beim bisherigen Gesamtverlust; eine neue unvalidierte Spielbarkeitskennzahl ersetzt ihn nicht.

Optional in Epoche 1, alle fünf Epochen und zum Ende/Frühstopp: bis zu drei feste, nach Map-ID-Hash ausgewählte **unterschiedliche Validierungssongs**, jeweils maximal 256 frei generierte Objekte. Nach Laden des besten Gewichtsstands wird entsprechend auf bis zu drei separaten Testsongs geprüft (ohne Testsongs: Validierung, ausdrücklich gekennzeichnet). Testdaten beeinflussen weder Lernrate noch Epochenauswahl. Berichte enthalten Einzelwerte, Mittelwerte, fehlgeschlagene Generierungen, zeitliche Überlappungen, Korrekturanteile, Sprungweiten-/Geschwindigkeitsquantile und Übereinstimmung mit Referenzzeitpunkten. Die feste Auswahl und der Seed erlauben Vergleiche; andere gültige Maps sind möglich. Diese begrenzte Präfixdiagnostik ist keine Spielspaßbewertung. Ein kompletter Song-/Genre-Benchmark und menschliche Spieltests stehen noch aus.

## Generierung und Spielbarkeit

1. Audiofenster verarbeiten, dabei die Songübersicht verwenden.
2. Konsistente gelernte Beat-Pulse und Beatlängen zu Tempoabschnitten verbinden. Phase und Tempo werden über mehrere Pulse angepasst, damit Frame-Rundung über Minuten keine Drift aufsummiert. Bei unsicherem Timing folgt eine konstante Librosa-Schätzung; alternativ manuelle BPM/Offset.
3. Onset-Maxima anhand gelernter lokaler Dichte und gewünschter mittlerer Dichte auswählen. Digitale Stille bleibt leer, auch nach Snapping. Viersekunden-Budgets und mindestens 35 ms Abstand begrenzen die Kandidatenzahl. Diese Regeln ersetzen kein perfektes Phrasen-Mapping.
4. Nacheinander vollständige Objekte erzeugen. Standardmäßig zwölf gemeinsame Geometriekandidaten prüfen, einschließlich unveränderter, wahrscheinlicher Mischkomponenten. Slider-Variation nutzt gemeinsame Translation, Rotation und gleichmäßige Skalierung einer gelernten Kurve. Unabhängiges Rauschen auf allen Kontrollpunkten wurde entfernt: Es konnte aus einer glatten Vorhersage Schleifen und Haken erzeugen. Modellstreuung und Temperatur steuern begrenzte Variation, unterschiedliche Kurvenformen kommen aus den gelernten Mischkomponenten. Bewertung kombiniert Modellwahrscheinlichkeit, erwartete Bewegung, Kontextbewerter, nötige Korrekturen und Lesbarkeit. Es ist lokale Kandidatenauswahl, keine globale Strahlensuche über alle möglichen Map-Fortsetzungen.
5. CS-gerechte Ränder durch gemeinsames Verschieben/Skalieren der Kontrollpunkte sichern; punktweises Abschneiden würde die Kurve verändern. Großzügige Notfallgrenzen für Sprünge, gültige Slider-Geschwindigkeit, passende Beat-Dauer und mindestens 35 ms bis zum nächsten Objekt bleiben erhalten. Eine zusätzliche Prüfung an 65 Kurvenpunkten verwirft Selbstkreuzungen, scharfe Haken und starke Umwege (Pfadlänge über 3,5-mal Start-Ziel-Abstand ab 20 Pixel Pfadlänge). Doppelte benachbarte Kontrollpunkte werden ausgeschlossen, da sie im `.osu`-Format Bézier-Segmente trennen und nicht dieselbe Bedeutung wie in einer einzelnen SVG-Kubik haben. Normale Bögen und S-Kurven bleiben möglich. Gibt es keinen geeigneten Sliderkandidaten, wird ein Circle erzeugt und als Korrektur gezählt. Absichtliche Stacks bleiben möglich; teilweise überdeckte Kreise werden weich bestraft. Nachfolgeobjekte sehen das endgültige Slider-Ende einschließlich Repeat-Parität.
6. Deterministischer Writer erzeugt `.osu`: kubische Kurven, Wiederholungen, New Combos, variable rote Timingpoints und passende grüne SV-Punkte. Anschließend erneut parsen und Dauern/Überlappungen prüfen, `.osz` mit Ogg-Audio schreiben. Das Modell lernt keine Dateisyntax.

Keine Garantie für Ranked-Kriterien; das Star Rating der fertigen Map wird bei verfügbarem Rechner offiziell gemessen. Slider-Kollisionen/visuelle Mehrdeutigkeit über ganze Pfade, präzise Stacking-Semantik und langfristige Ermüdung sind nicht vollständig modelliert. Die Webvorschau verwendet Kurven, Wiederholungen, AR und CS, ist aber kein identischer osu!-Renderer. Hitsounds, Storyboard, Kiai, Spinner und SliderTickRate werden nicht gelernt; TickRate ist beim Export 1.

Die Kurvenprüfung ist eine konservative Stil-/Spielbarkeitsregel dieses Generators, keine offizielle osu!-Zulässigkeitsregel. Absichtlich verschlungene Spezialslider können dadurch wegfallen; die diskrete Prüfung ist kein mathematischer Beweis gegen jede denkbare Kurvenentartung. Der Bericht ergänzt `slider_shape_issues` und `slider_detour_p95`. Netzwerk, Trainingsziele und Gewichtsformat sind durch diese Decoderkorrektur unverändert. Auch vorhandene v3-Gewichte profitieren beim erneuten Generieren; bereits exportierte Maps werden nicht verändert. Referenz für Slider-Segmentierung: [offizielle Dateiformat-Dokumentation](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29#sliders).

## Dateien und Wiederaufnahme

`model.safetensors`: bester Inferenzstand. `training.safetensors`: letzter Optimierer-/Gewichtsstand, inklusive Schritt-/Epochennummer als Safetensors-Metadaten. Alle Komponenten des Generators bleiben in einer Gewichtsdatei. Der Optimierer ist nur für Training nötig.

Checkpoints nach dem ersten Update, alle 500 Updates und an Epochengrenzen; Dateien werden atomar ersetzt. Ein Abbruch mit mindestens einem erfolgreichen Update erhält den Lauf. Nach hartem Prozessabbruch registriert der nächste App-Start `recovery.json`-Stände. Ein angefangener Epochenteil kann erneut gesehen werden. Resume verlangt denselben Datensatz samt unverändertem Inhaltsfingerprint; Modellgröße/Split werden übernommen, Zufallsfolge und Lernratenplan nicht exakt rekonstruiert. Für neue Songs im Trainingsdatensatz ein neues Modell anlegen. Download-ZIP enthält nur Inferenzgewichte und JSON, keinen Optimierer; lokales `data` für vollständige Sicherung kopieren.

Audio-/Label-Caches sind atomare NumPy-Dateien ohne Pickle und werden mit Memory-Mapping gelesen. Große Sammlungen brauchen weiterhin Cache-Platz und Vorbereitungszeit. FastAPI bedient React, ein Worker arbeitet Jobs seriell, SQLite speichert Status und portable `data://`-Pfade. Modellimport erlaubt nur bekannte Architektur, geprüfte Konfiguration, endliche Float32-Safetensors und JSON. Keine Ausführung heruntergeladener Modellprogramme.

## Fachliche Referenzen

- [Dance Dance Convolution](https://proceedings.mlr.press/v70/donahue17a.html): Inspiration für Audio-zu-Rhythmus und bedingte Objektfolgen.
- [osu! Difficulty Preprocessing](https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Osu/Difficulty/Preprocessing/OsuDifficultyHitObject.cs): Orientierung an Sprungdistanz, verfügbarer Zeit, Winkeln und Slider-Reisewegen. Unsere Größen sind nicht die offiziellen Skill-/Star-Formeln.
- [osu!-Dateiformat](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29): Timing, Slider und deterministischer Export.
- [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator): Referenz für umfangreichere Audio-/Ereignismodelle; keine übernommenen Gewichte und kein Anspruch auf gleichwertige Ergebnisse.
- [DirectML-Operatorübersicht](https://github.com/microsoft/DirectML/wiki/PyTorch-DirectML-Operator-Roadmap): Plattformverträglichkeit zusätzlich auf echter Hardware geprüft.
