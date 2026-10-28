# Architekturvergleich: Evidenz und Prüfprotokoll

Stand: 28. Oktober 2026. Fortsetzungsplan: [architecture-roadmap.md](architecture-roadmap.md).
Diese Messungen trennen technische Gültigkeit, Ähnlichkeit zu einer Referenzmap und
musikalische Qualität. Die letzte Kategorie benötigt menschliche Spieltests.

## Aufgabe 1: feste Baseline und beobachtbare Entscheidungen

Hypothese: Die bisherige Begrenzung auf bis zu 256 Köpfe am Songanfang verdeckt
spätere Pausen, Tempoabschnitte und Fensterübergänge. Zudem durfte die letzte
vorhergesagte Kopfzeit den Referenzhorizont bestimmen. Eine sparsamere Ausgabe
konnte dadurch einen anderen Abschnitt bewerten.

Baseline vor Änderungen am Generierungscode:

- Git-Stand `4e349d11910e273a6bedbf46602c6afe11abfc90`.
- BobV1.5, Modell-ID `a35a11802f7a473eb0629074d9b0a4a4`, Phrase Mapper v3,
  großes Modell, beste Epoche 17, 716 vorbereitete Maps.
- SHA-256 der unveränderten Safetensors-Datei:
  `196df1447b3abdd8ea98cd9e65977737bcd22dedfa20ef7f8f1c49fd5001e313`.
- Dataset-Signatur `03ec22d8d3007d30f0fa2b5db5832945f92b4b625c03014d8c5e6bc3f58b5464`.
- RTX 5070 Ti, 16 GiB; Python-/Torch-Version, Dateihashes, Einstellungen und
  Laufzeiten stehen im lokalen Bericht `data/benchmarks/architecture/baseline.json`.
- Sechs ausdrücklich als Validierung ausgewiesene Audios aus dem vorhandenen
  32-Song-Benchmark. Keine neuen Tests auf bisherigen Abschlusstestgruppen.
- Vollständige Rhythmusvorhersagen, Mel-Daten, Ziele, Timing und gewählte Zeiten
  für alle sechs Songs gesichert; drei feste Generierungspräfixe und ein
  vollständiger 46,79-Sekunden-Song für die Seeds 42 und 73.
- Temperatur 0,8, zwölf Kandidaten, Snapping aktiviert; die acht globalen
  Conditions entsprechen jeweils der vorbereiteten Referenzmap. Somit keine
  Aussage über unabhängige, vom Nutzer vorgegebene Stile.

Einige frühere Cachedateien waren nicht mehr vorhanden. Die zugehörigen originalen
Maps wurden per SHA geprüft, deren Ziele im isolierten Benchmark neu berechnet.
Fehlende Mel-Caches wurden aus vorhandenem Audio erneut berechnet. Die produktiven
Gewichte, Modelle und Datensätze wurden nicht überschrieben.

Prüfkriterien: Trace an/aus liefert bei gleichem Seed dieselben Objekte;
Auswertungsgrenzen werden vor der Generierung festgelegt; auch späte Abschnitte
werden geprüft; Abbruch bleibt sichtbar; Export wird erneut geparst.

Umgesetzt in `rollout_sections.py` und `rollout_evaluation.py`:

- Feste 24-Sekunden-Proben für Anfang, Mitte, Ende, niedrige/hohe Audioaktivität
  und einen zentralen Audiofensterübergang, mit zusammengeführten Doppelproben.
  Audioaktivität wird ausdrücklich nicht als Refrain- oder Breakannotation
  ausgegeben. Bestehende digitale Stille ist von beliebig leisen Ausklängen zu
  unterscheiden; diese Proben identifizieren keine Instrumente.
- Eigene Objekthistorie ab Abschnittbeginn, volles Audio und globale Songübersicht;
  der echte nächste Kopf bleibt auch außerhalb des ausgewerteten Bereichs für
  Sliderplanung und Decoderquery verfügbar.
- Höchstens 128 Objekte pro Abschnitt. `truncated` und vorab festgelegte Grenzen
  sind im Bericht sichtbar; der Referenzumfang wird nicht auf das erzeugte Ende
  verkürzt. Slideranteil und Sprungweiten der Referenz verwenden denselben Bereich.
- Nach Laden der besten Gewichte zusätzlich höchstens ein kompletter Song mit
  maximal 300 Sekunden und 2048 Objekten. Längere Songs werden nicht als komplette
  Läufe ausgegeben. Eine Objektbegrenzung verhindert `completed_full_song=true`.
- Wieder eingelesener Export prüft Anzahl sowie Dauer- und Endpunktabweichungen.
  Die Berichte bleiben von der Checkpoint-Auswahl getrennt.

Ergebnisse: Alle acht ursprünglichen Baselineläufe konnten exportiert werden,
mit gleicher Objektzahl nach erneutem Parsen. Größte Dauerabweichung 0,002061 ms.
Generierung 4,26–7,46 s pro Lauf; der erste gesonderte Rhythmuspass brauchte
40,86 s einschließlich Anlaufkosten, weitere 0,34–0,88 s. Das ist keine isolierte
Hardwareleistung. Ein späterer Smoke-Test der erweiterten Diagnostik lieferte
15 erfolgreiche Proben auf drei Songs einschließlich eines vollständigen Songs;
er lief auf dem sich ändernden Arbeitsstand und ist kein isolierter A/B-Vergleich.

Offen: manuell markierte musikalische Abschnitte, umfassende Stilabdeckung und
Langsong-Auswertung mit längerem freien Verlauf vor den bewerteten Abschnitten.
Abschnittsneustarts allein messen keine über Minuten kumulierten Kontextfehler.

## A/B mit identischen v3-Gewichten

`data/benchmarks/architecture/inference-b.json` hält den eingefrorenen Stand nach
Aufgaben 2 und 3 fest. Er enthält den korrigierten Rhythmusselektor einschließlich
isolierter starker Akzente und Beatphasen-Konfidenz. Die abschließende Fassung
unterscheidet auch 1/12- und 1/16-Familien, behandelt innerhalb einer halben
Audioframeauflösung mehrdeutiges Snapping zugunsten der gröberen passenden
Unterteilung und zentriert bedingte Dauer-/Repeat-Scores nach deren Entropie.
Frühere B-Läufe sind unter `superseded-inference-b-*` archiviert und gehören nicht
zu dieser Tabelle. B verwendet dieselben Gewichte, Seeds und globalen Einstellungen wie A.
Bei Präfixen definiert das letzte A-Objektende plus 35 ms einen festen gemeinsamen
Zeitraum; der vollständige Vergleich umfasst den ganzen Song.

| Song-ID (gekürzt), Bereich | A-Köpfe | B-Köpfe | A Referenz-F1 | B Referenz-F1 |
| --- | ---: | ---: | ---: | ---: |
| `1d0a5886`, Präfix | 256 | 250 | 0,7294 | 0,7183 |
| `23aa63b5`, Präfix | 256 | 255 | 0,7436 | 0,6784 |
| `e4c24ae0`, vollständig | 176 | 164 | 0,8138 | 0,7953 |

Die Kopfzahlen und F1-Werte sind für Seeds 42 und 73 gleich; Geometrie bleibt
seedabhängig. **Die Referenzähnlichkeit sinkt in diesen drei Fällen.** Weniger
Köpfe oder weniger Reparaturen belegen keine bessere musikalische Interpretation.
Aufgaben 2 und 3 wurden hier gemeinsam verändert; der gesonderte Rhythmusvergleich
steht in `scripts/benchmarks/rhythm_selection.py` und im Roadmap-Bericht.

Alle acht B-Läufe erfolgreich, keine vom Generator gemeldeten zeitlichen
Überlappungen; maximale Export-Dauerabweichung 0,001380 ms. B-Laufzeiten lagen bei
4,065–6,687 s nach dem ersten Lauf von 10,542 s. Einer der acht Läufe aktivierte
zusätzlich den Trace. Die ursprüngliche Baseline lief unter anderer
Hintergrundlast; daraus folgt kein sauberer Durchsatzfaktor gegenüber A.
Ein erneuter vollständiger Lauf mit identischen Eingaben ohne Trace produzierte
exakt dieselben 164 Objekte wie mit Trace; 9298
Traceereignisse sind unter `e4c24ae07fb5530f019ae28b-42-full-B-trace.json` gesichert.

## Aufgabe 9: Korpus, Dubletten und reservierter Abschlusstest

`scripts/benchmarks/corpus_inventory.py --scan-maps` liest SQLite ausschließlich
über `mode=ro`. Ein Inventar wird getrennt vom produktiven Katalog gespeichert.
Songgruppen verwenden Audio-SHA oder normalisierte Künstler-/Titelangaben und
erhalten jetzt inhaltsbasierte IDs statt von der Datensatzreihenfolge abhängiger
Union-Find-Indizes. Das kann neue Splits bewirken und ist eine beabsichtigte neue
Datengrundlage; historische Exposures werden über Audiohashes verfolgt.

Lokales Inventar:

- 3348 Maps, 790 unterschiedliche Audiohashes, 739 heuristische Songgruppen.
- 3346 Maps erfolgreich geparst, zwei mit zu vielen Sliderkontrollpunkten
  ausgeschlossen. Zusammen 1.257.101 Circles/Slider und 530.218 Slider.
- 15.438 Slider überschreiten unterstützte Geometrie-/Dauergrenzen; das ist eine
  Mapanalyse, keine vollständige Trainingsmaskenprüfung gegen Audiodauer/-bounds.
  Die sechs vollständig vorbereiteten Diagnosemaps enthalten elf maskierte Slider.
- 6808 Spinnerintervalle; 811 Maps mit mehreren roten Timingpoints. Taktarten
  laut Importtext: 3301 Maps mit Meter 4, 136 mit 3, 41 mit 5, 42 mit 6, 18 mit 7.
  Eine Map kann mehrere Meter enthalten; diese Zählwerte sind nicht exklusiv.
- 3348 bekannte gespeicherte Star-Ratings von 0,71 bis 54,81. Das sind vorhandene
  Bibliothekswerte, keine neu berechneten Ratings und keine verlässlichen Stilziele.
- Der erste Spektralvergleich über sechs Audios fand keine Hinweise. Der
  anschließende vollständige Development-Scan versucht alle790 Audiohashes und
  analysiert785 (99,37%).236 vorhandene Mel-Caches wurden nur gelesen,
  549 im isolierten Benchmarkruntime neu berechnet. Fünf Audios bleiben offen:
  vier außerhalb der unterstützten Dauer von1s–20min, eines nicht dekodierbar.
  Es gibt keine Hashabweichungen. Laufzeit194,38s mit zwei CPU-Workern.
- Elf konservative Ähnlichkeitshinweise (Cosine mindestens0,995, relative
  Dauerabweichung unter1%), davon vier zwischen bisher getrennten Songgruppen:
  Snow Goose/Stream Practice Maps, Harumachi Clover Swing Dictate Edit/Swing,
  and I'm home mit umsortierten Künstlernamen, Square Jump Practice Maps/cheatreal.
  Das sind **Prüfkandidaten, keine bewiesenen Dubletten**. Keine automatische
  Zusammenführung oder Splitänderung. Der Abgleich mit dem C-Manifest findet
  keinen Hinweis zwischen dessen Training und Validierung; Nichtfinden beweist
  keine vollständige Dublettenfreiheit, insbesondere nicht für Teilausschnitte.

Vor diesem erweiterten Scan festgelegte Hypothese: Unterschiedliche Hashes und
Metadatennamen können dieselbe Aufnahme verbergen. Baseline sind die sechs
vorbereiteten Audios; Kriterium ist breite, explizit ausgewiesene Abdeckung und
reproduzierbare Hinweise ohne Eingriff in Splits. `scripts/benchmarks/acoustic_inventory.py`
prüft nur ausdrücklich als Development reservierte IDs und sperrt vollständige
verbundene Holdoutgruppen vor jedem Datei-/Cachezugriff. Neun fokussierte Tests
prüfen das auch mit künstlichen Reservierungen, Aliasnamen, geänderten Hashes
und beschädigten Caches. Im tatsächlichen Bestand sind keine frischen Gruppen
reserviert. Der Bericht `acoustic-inventory.json` enthält Quellen-/Audiohashes,
kompakte Signaturen, Fehler und Kandidaten; `acoustic-split-check.json` den
separaten Abgleich zum C-Manifest. Signaturen sind weder Genrelabels noch ein
Beweis gleicher musikalischer Struktur. Bericht-SHA256:
`aecabefd5341aff796ec5a71f3303d6f466e3512ec929c0782aec33b9d5aaab0`.

**Kein frischer Abschlusstestsatz verfügbar:** Alle 739 Songgruppen sind bereits
in der konservativen Vereinigung vorhandener Modell- und Datensatzexposures
enthalten, einschließlich früherer Validierungs-/Testnutzung. Die Datei
`holdout-reservation.json` weist deshalb ausdrücklich
`unavailable_all_groups_previously_exposed` und null reservierte Gruppen aus.
Die vorhandenen Tests als unberührt umzubenennen wäre falsch. Neue unabhängige
Songs oder ein nachweisbar bislang ungenutzter Korpus sind für einen endgültigen
Bestätigungstest nötig. Ein bestehender Reservierungsplan wird vom Werkzeug
nicht stillschweigend neu gemischt.
Der Nutzer hat bestätigt, dass kein weiterer ungenutzter Ordner vorliegt und er
neue Songs künftig selbst zusammenstellt. Dies blockiert die übrige Entwicklung
nicht; ein unabhängiger Abschlusstest wird ausdrücklich nicht vorgetäuscht.

Metadaten-Gruppierung kann unterschiedliche Aufnahmen zusammenfassen; umgekehrt
bleiben anders benannte, neu codierte Dubletten möglich. Genre, Gesang und
Instrumente sind nicht annotiert. Messbare Dichte-, Abstands-, Pausen- und
Slidermerkmale werden daher nur als unsichere Proxys geführt. Mehrere
Schwierigkeiten desselben Songs erhöhen die musikalische Vielfalt nicht.

## C: gleichbudgetierte Trainingsablationen mit neuer Repräsentation

`c-training-records.json`: 19 Trainings- und fünf Validierungssongs aus den
ursprünglichen 32, mit Map-/Audioidentitäten und optionalen Mel-Pfaden.
Acht Songs wurden wegen früherer Testzuordnung ausgeschlossen, anhand der
Vereinigung aller Modell-Testhashes und des früheren Trainingssteuerungsvergleichs.
Der neue Manifest enthält keine v3-Objektziele: Diese werden mit der neuen
Repräsentation neu erzeugt. Vergleichsarme benötigen gleiche Songs, Kapazität,
Seeds und Budget. Kurze C-Läufe lassen sich nicht fair gegen das wesentlich
länger trainierte v3-Modell nach Gesamtqualität ordnen.

Die zehn Läufe sind abgeschlossen: fünf Varianten mit Seeds 42/73, jeweils acht
Epochen und 176 Optimizerschritte (22 je Epoche), Batch 8, Lernrate 0,0003,
Weight Decay 0,01, BF16 auf RTX 5070 Ti. Alle instanziieren dasselbe große Modell
mit 33.451.382 Parametern und demselben Startzustand pro Seed. Interne
Ablationsschalter maskieren musikalische Zeit beziehungsweise Planprojektion
und Planverlust. Die Datenaufbereitung erzeugt neue Targets aus Originalmaps;
kein alter Checkpoint wurde konvertiert. Audioaugmentation und Timingstörungen
sind in allen Armen gleich. CUDA ist nicht als bitweise deterministisch zugesichert.
Gleiche allokierte Parameter bedeuten bei deaktivierten Pfaden keine exakt
gleiche aktive Kapazität. Neun von zehn besten Epochen liegen am Budgetende;
der Versuch vergleicht begrenzte Lernbudgets, keine auskonvergierten Modelle.

Die Änderungen werden paarweise getrennt:

| Variante | Vergleich | Einzige zusätzliche Versuchsänderung |
| --- | --- | --- |
| `seconds` | neue kontrollierte Basis | Sekundenqueries, kein Plan, historische Shift7-Kritikersupervision nur im Benchmark |
| `time` | `seconds` | elf musikalische Zeitkanäle |
| `plan` | `time` | sieben vorhergesagte Planproxies und Hilfsverlust |
| `critic` | `plan` | strukturgleiche plausible Gegenbeispiele mit weicher Präferenz |
| `context` | `plan` | begrenzte eigene Positionshistorie, ausschließlich Strukturziele im Zusatzpfad |

`context` vergleicht sich ausdrücklich mit `plan`, nicht mit `critic`; die beiden
Eingriffe aus Aufgabe 8 wurden nicht gemeinsam als ein Experiment ausgegeben.
Eigenkontext startet nach dem Warmup und erreicht innerhalb dieses Budgets die
Wahrscheinlichkeiten 0,02–0,08 in Epochen 5–8. Das bleibt ein kurzer Entwicklungsversuch.
Trotz gleicher Seeds unterscheiden sich `context` und `plan` bereits vor dem
aktivierten Zusatzpfad geringfügig (Seed 42, Epoche 1: 2,7699 gegenüber 2,7637).
Die nichtdeterministischen CUDA-/BF16-Pfade wurden dafür nicht weiter isoliert.
Die Versuche teilen Initialisierung und Datenplan, sind jedoch keine bitgenau
gepaarten Trainingsverläufe. Kleine Effekte aus zwei Seeds erlauben keine
belastbare Kausalrangfolge; der maximale Kontextanteil 0,20 wurde nicht erreicht.

Alle Checkpoints werden nach derselben aktuellen Summe der zwölf Hauptverluste
auf fünf Validierungssongs gewählt. Plan- und Kontexthilfsverlust beeinflussen
diese Auswahl nicht. Die neue weiche Kritikersupervision kann dennoch den
Kompatibilitätsterm anders kalibrieren; deshalb ist die Summe ohne diesen Term
zusätzlich nötig. Sie ist ebenfalls nur Referenznähe, kein menschliches Urteil.

| Arm | Seed | beste Epoche | Hauptverlust | ohne Kritikerterm |
| --- | ---: | ---: | ---: | ---: |
| seconds | 42 | 8 | 1,883939 | 1,764589 |
| seconds | 73 | 7 | 1,810017 | 1,695275 |
| time | 42 | 8 | 1,900795 | 1,781245 |
| time | 73 | 8 | 1,831923 | 1,710875 |
| plan | 42 | 8 | 1,947176 | 1,827873 |
| plan | 73 | 8 | 1,854973 | 1,734094 |
| critic | 42 | 8 | 1,975377 | 1,907478 |
| critic | 73 | 8 | 1,800234 | 1,732338 |
| context | 42 | 8 | 1,867187 | 1,744787 |
| context | 73 | 8 | 1,844879 | 1,724768 |

Mittlere gepaarte Änderungen ohne Kritikerterm: Zeit **+0,016128**, Plan
**+0,034923**, neuer Kritiker **+0,038924**, Eigenkontext **−0,046206**; kleiner
ist besser. Die ersten drei Erweiterungen verbessern diese kurze
Teacher-Forcing-Probe somit nicht. Der scheinbare Gesamtverlustgewinn des neuen
Kritikers von 0,013269 entsteht durch einen um 0,052193 kleineren Kritikerterm;
die übrigen Hauptziele werden zusammen schlechter. Das ist ausdrücklich kein
Qualitätsbeleg und keine Grundlage für eine neue Checkpointmetrik.

Die zunächst fest geplanten neun freien Abschnitte je Lauf liefern ebenfalls
gemischte Ergebnisse. Mittlere gepaarte F1-Änderungen: Zeit −0,007145,
Plan +0,011968, Kritiker −0,004835 und Eigenkontext −0,003115. Damit geht der
Teacher-Forcing-Gewinn des Eigenkontexts nicht mit besserer freier Referenz-F1
einher. Alle 90 Läufe exportieren erfolgreich ohne Trunkierung, erkannte
Zeitüberlappung oder Verletzung der konservativen Formpolicy; größte
Dauer-/Endpunktabweichung nach erneutem Parsen 0,0007562 ms/0,0000500 px.
Die folgende erweiterte Auswertung wird getrennt von diesen anfänglichen neun
Proben je Arm berichtet.
Alle ersten 90 freien Läufe verwenden `audio-fallback`, jeweils ein konstantes
Timingsegment und unbekannte Metrik. Die Teacher-Forcing-Validierung verwendet
dagegen Referenztiming. Die Trainingsstörungen verringern diese Lücke, beseitigen
sie aber nicht. C belegt somit keine gelungene automatische Erkennung von
Tempoänderungen, Metrik oder Downbeats. Dafür existieren technische kontrollierte
Tests, weiterhin keine belastbare musikalische Auswertung auf neuen Songs.

Gesamtdauer einschließlich Training, Validierung und erster freier Proben:
569,52 s für zehn Läufe. Der erste `seconds-42`-Lauf enthält Anlaufkosten
(82,42 s gegenüber 51,69 s bei `seconds-73`); daraus lässt sich keine
Architekturgeschwindigkeit ableiten. Die reinen Trainingszeiten der übrigen
Arme liegen bei 23,43–28,68 s. Separate Eigenkontextpfade benötigen erwartbar
zusätzliche Rechenarbeit; gleiche Updatezahl ist kein exakt gleicher FLOP-Budget.

Reproduktion: `scripts/benchmarks/architecture_training.py` prüft Manifest,
Code, Modellkonfiguration, Budget und Gewichtsdateien vor Wiederverwendung.
`data/benchmarks/architecture/training/report.json` enthält alle Epochen,
Teilmetriken, Einstellungen und Laufzeiten; je Arm liegen Gewichte und Bericht
im eigenen Unterordner. Der eingefrorene Quellstand liegt unter `training/source`.
Daten-SHA256: `9457708be6dc921188ea58e686bd8a2a8a5ef1d0cc47d05d825a599baf80c2fd`;
Quell-SHA256: `19eeea71fe3e25918b8cc987f62a1bf3fc095d0ddc15543aa9e476c73dc627f6`.
Er enthält den Stand vor der abschließenden Korrektur des technischen Timings
vor dem ersten positiven manuellen Timingpoint; C verwendet geschätztes Timing.
Spätere Formatierung und strengere Metadatenprüfung verändern diesen Snapshot
nicht. Die zehn Versuchsgewichte sind nicht als produktive Studiomodelle registriert.

### Erweiterte freie Generierung mit C-Gewichten

`scripts/benchmarks/architecture_rollouts.py` und
`training/extended-rollouts.json` verwenden dieselben eingefrorenen Quellen
und zehn Gewichtsdateien. Vorab festgelegt sind alle 25 unterschiedlichen
24-s-Proben über die fünf Validierungssongs und zusätzlich der kürzeste ganze
Song (Despacito, 46,78 s). Pro Abschnitt höchstens 128, pro Vollsong 2048 Objekte;
Lookahead und Motivoption bleiben aus. Die 90 bereits identischen Proben werden
hashgeprüft übernommen und als solche gekennzeichnet; 170 Läufe kommen neu hinzu.

**260/260 Läufe erfolgreich**, darunter 250 Abschnitte und zehn vollständige
Songläufe, keine Trunkierung, erkannte Überlappung oder konservative Formverletzung.
Maximale Exportabweichung 0,0011328 ms bei der Dauer und 0,0000500 px beim Endpunkt.
Alle zehn Vollsongs melden `completed_full_song=true`; das ältere `scope`-Feld
im Snapshot nennt wegen der weitergereichten Bereichsgrenzen auch dort `section`.
Der produktive Bericht korrigiert diese Bezeichnung; Entscheidungen und
ausgewerteter Bereich ändern sich dadurch nicht.

| Arm | Seed | F1 der 25 Abschnitte | Köpfe/Slider in Abschnitten | mittlere Geometriereparatur | Vollsong-F1 | Vollsong Köpfe/Slider |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seconds | 42 | 0,651790 | 1894/379 | 0,000878 | 0,815951 | 153/55 |
| seconds | 73 | 0,600626 | 1663/286 | 0,000947 | 0,736196 | 153/32 |
| time | 42 | 0,625063 | 1611/493 | 0,000789 | 0,797468 | 143/72 |
| time | 73 | 0,593836 | 1624/739 | 0,001672 | 0,749226 | 150/79 |
| plan | 42 | 0,635260 | 1634/300 | 0,000757 | 0,802589 | 136/47 |
| plan | 73 | 0,618008 | 1614/543 | 0,001669 | 0,785276 | 153/84 |
| critic | 42 | 0,623371 | 1597/457 | 0,000765 | 0,790997 | 138/75 |
| critic | 73 | 0,615948 | 1707/796 | 0,001233 | 0,789157 | 159/80 |
| context | 42 | 0,637340 | 1875/510 | 0,000751 | 0,786787 | 160/64 |
| context | 73 | 0,619324 | 1648/332 | 0,003547 | 0,788991 | 154/18 |

Die mittlere Abschnitts-F1 beider Seeds ist 0,626208/0,609450/0,626634/0,619660/
0,628332 für seconds/time/plan/critic/context. Gepaarte Änderungen: Zeit −0,016758,
Plan +0,017184, Kritiker −0,006974, Kontext +0,001698. Die kleine Kontextänderung
wechselt gegenüber den anfänglichen neun Abschnitten sogar das Vorzeichen.
Das Ergebnis hängt sichtbar vom Umfang der Probe ab. Die großen Unterschiede
im Slideranteil zwischen Seeds sind ebenfalls kein Nachweis beabsichtigter
musikalischer Vielfalt oder eines konsistenten Mappingstils.

Aufschlüsselung pro Song, Mittel über fünf Bereiche und beide Seeds:

| Song (Map-ID gekürzt) | seconds | time | plan | critic | context |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kyoumei No True Force (`23aa63b5`) | 0,713836 | 0,699829 | 0,715389 | 0,720139 | 0,683857 |
| Despacito (`e4c24ae0`) | 0,773902 | 0,772382 | 0,793391 | 0,788124 | 0,793741 |
| Nakimushi Girl (`7be6b286`) | 0,526545 | 0,512635 | 0,523834 | 0,515256 | 0,541391 |
| unfinished TV (`36e10659`) | 0,504574 | 0,482273 | 0,492755 | 0,480362 | 0,498813 |
| Sakura Sweet Kiss (`fa5d9bc7`) | 0,612183 | 0,580131 | 0,607800 | 0,594419 | 0,623858 |

Kontinuierliche Referenzstilproxies in derselben Reihenfolge: mittlerer
Slideranteil 0,403/0,338/0,386/0,609/0,506 und Mittel der abschnittsweisen
Jump-P95 von 258/268/239/170/159 Pixeln. Diese Werte liefern Kontext für
Unterschiede, bilden jedoch keine gesicherten Jumps-, Tech- oder Readingklassen.
Insbesondere verschlechtert Eigenkontext Kyoumei gegenüber Plan um 0,031532 F1,
während es auf mehreren anderen Songs gewinnt. Ein Gesamtmittel verdeckt das.

| Bereichslabel | seconds | time | plan | critic | context |
| --- | ---: | ---: | ---: | ---: | ---: |
| Anfang | 0,639257 | 0,629276 | 0,629454 | 0,621599 | 0,641727 |
| Mitte | 0,578062 | 0,558848 | 0,579904 | 0,576730 | 0,584315 |
| Ende | 0,646755 | 0,636074 | 0,657946 | 0,650371 | 0,652487 |
| niedrige Aktivität | 0,631506 | 0,620488 | 0,632896 | 0,626430 | 0,639107 |
| hohe Aktivität | 0,671649 | 0,655413 | 0,667869 | 0,664066 | 0,671154 |
| Fensterübergang | 0,595317 | 0,567639 | 0,597996 | 0,585534 | 0,591976 |

Bereiche können überlappen; niedrige Aktivität teilt hier jeweils einen Bereich
mit Anfang oder Ende. Die Labels sind keine unabhängigen Beobachtungen und
keine Annotation von Refrains/Breaks. 260 Auswertungen bleiben fünf Songs und
zwei Seeds, nicht 260 unabhängige Musikbeispiele. Sämtliche freien Läufe nutzen
weiterhin konstanten Audio-Fallback. Alle fünf Validierungsmaps haben Meter 4;
nur Despacito enthält zwei rote Punkte mit verschiedenen Beatlängen
(659,341/674,157 ms). Unter 19 Trainingsmaps haben drei mehrere rote Punkte,
zwei verschiedene Beatlängen und nur eine auch Meter 3. Die begrenzte Vielfalt
erlaubt keinen empirischen Nachweis der Behandlung anderer Taktarten.

Die Erweiterung benötigt 507,54 s Wandzeit; 461,50 s entfallen auf neu gemessene
Generierungen. 276,21 s der übernommenen früheren Proben wurden **nicht** erneut
verbraucht. Anlaufkosten und Auswertungsumfang verhindern eine reine
Architekturdurchsatzrangfolge. Die JSON-Berichte enthalten für jede Probe
Settings, Hashes, Grenzen, Referenzproxies, Reparaturen und Laufzeiten.

## Verblindete menschliche A/B-Spieltests

`scripts/benchmarks/blind_playtest.py` erstellt aus expliziten A/B-Paaren anonyme
`.osz`-Archive, einen öffentlichen Ablaufplan und einen Fragebogen. Eine getrennte
Datei außerhalb des Teilnehmerordners hält die Zuordnung der Bedingungen.
Metadaten werden neutralisiert; erneut geparste Objekte und Timing müssen vor
und nach der Anonymisierung identisch sein. Präsentationsreihenfolge und A/B-
Zuordnung sind per Seed reproduzierbar.

Sechs lokale Paare des endgültigen B-Stands (drei Songs, zwei Seeds) sind unter
`data/benchmarks/architecture/blind-playtest` vorbereitet. Der Fragebogen enthält
Timing/Rhythmus, musikalische Interpretation, Motive/Übergänge, Lesbarkeit,
Ermüdung und Spielspaß. Präferenz, Unsicherheit und konkrete Audiozeitstellen
werden getrennt erfasst. Alle Antwortfelder sind leer; **menschliche Bewertungen
wurden nicht erhoben**. Der private Schlüssel wird Teilnehmenden nicht gegeben.

Wiederholbare Erfassung neuer Baselines: `scripts/benchmarks/architecture_baseline.py`
verlangt Modellverzeichnis, versionierten Prepared-Manifest und ein frisches
Ausgabeverzeichnis. Quellstand, Gewichte, Daten, Einstellungen, Seeds und
Zeitgrenzen werden festgehalten. Gesicherte alte Versuchssourcen dienen nur der
Reproduktion der Evidenz, nicht als alternativer produktiver Inferenzpfad.

## Abschlussstand und nächste Priorität

Abschließende Prüfung: **198 Backendtests bestanden** (80,79s), einschließlich
API-Training/Generierung/Download, Schemaablehnung, Timing-/Slider-Roundtrips,
Kausalität, Masken, endlichen Gradienten, CUDA-Pfaden, Trace-Parität,
Motivtransformationen und Split-/Dublettenlogik. Ergebnisdatei:
`data/benchmarks/architecture/final-tests.xml`. Frontend: TypeScript/Vite-Build
und sieben Node-Tests bestanden. Ein nativer DirectML-Vorwärts-/Rückwärtsschritt
mit den neuen Trainingszweigen wurde zusätzlich geprüft; die gesamten
Trainingsablationen liefen auf CUDA.

Der erste Gesamtprüflauf deckte eine Testisolation auf: Absichtlich ungültige
Modellfixtures blieben im gemeinsamen Testkatalog und störten spätere API-Tests.
Sie werden jetzt in `finally` entfernt. Zusätzlich ignoriert die Modellübersicht
fehlende Architekturmetadaten sicher; die echte Generierung weist inkompatible
Artefakte weiterhin ausdrücklich zurück. Der anschließende vollständige Lauf
ist grün. Unter Windows benötigt FastAPI TestClient lokale Socketpaare, die in
der verwendeten Sandbox blockierten; API-/Gesamttests wurden deshalb außerhalb
dieser Sandbox mit weiterhin isoliertem `data/tests`-Verzeichnis ausgeführt.

Nachweisbare technische Verbesserungen: Entscheidungen sind bis zum erneut
geparsten Export nachvollziehbar; Trace an/aus erhält die Ausgabe. Feste
Viersekunden-Top-N-Budgets entfallen, schwache Peaks füllen keine Pflichtquote.
Typ, Dauer, Repeats und Geometrie werden gemeinsam verglichen, statt einmal
vorab einen Typ zu ziehen. Meter, unbekannte Metrik und manuelle Timingabschnitte
haben einen ausdrücklichen Vertrag. Kontextaugmentation verändert ganze
Geometrien; die alternative Historie erzwingt keine widersprüchliche nächste
absolute Referenzposition. Neue Modell-, Target- und Cacheformate werden
eindeutig geprüft. Inkompatible Modelle werden vor Geräte-/Audioladen mit einer
Neutraining-Anweisung abgewiesen. Vorhandene Dateien bleiben erhalten.

Der Hybridansatz bleibt bestehen: Rhythmus, Plan und Objektvorhersage gehören
weiterhin zu einem gemeinsam trainierten Modellartefakt. Auswahl und Guards
sind deterministische Regeln; `.osu`-Export und Kontrolle bleiben deterministisch.
Seeds erlauben unterschiedliche ganze Geometrieinterpretationen. Es wurden
weder zufällige Timingfehler noch zusätzliches unabhängiges Kontrollpunktrauschen
eingeführt. Die bestehenden acht Geometriemischkomponenten bleiben erhalten.

**Eine bessere musikalische Beatmapqualität ist nicht belegt.** A/B zeigt
niedrigere Referenz-F1, Lookahead ist langsamer und repariert in der Probe mehr,
die Motivzuordnung bleibt auf echtem Material wirkungslos, und die kurzen
Trainingsablationen ergeben kein einheitliches positives Bild. Geringere
Bewegung oder weniger komplexe Slider gelten hier nicht als Erfolg. Die neue
Repräsentation ist technisch nutzbar, braucht aber vollständiges Neutraining
und Bewertung auf ausreichend vielfältigem Material; die kleinen C-Checkpoints
sind keine empfohlenen fertigen Modelle.

Die nächste fachliche Priorität sind die bereits vorbereiteten verblindeten
Spieltests mit konkreten Zeitstellen und wenigen manuellen Annotationen zu
musikalischen Akzenten, Pausen und wiederkehrenden Phrasen. Sie können klären,
ob die negative Referenz-F1 gültige andere Interpretationen oder tatsächliche
Rhythmusverluste bedeutet. Danach lassen sich Rhythmusscore und gemeinsame
Sliderplanung gezielt untersuchen, statt weitere unkalibrierte Komponenten
hinzuzufügen. Für Motivzuordnung und semantische Skillset-/Layerziele fehlen
verlässliche Labels; Source Separation oder ein zusätzliches Fremdmodell sind
noch nicht begründet. Ein neuer, vorab reservierter Songkorpus ist Voraussetzung
für den unabhängigen Bestätigungstest. Neue Architekturentscheidungen dürfen
diesen Testsatz nicht schrittweise zu einem weiteren Entwicklungsset machen.

Alle Aufgaben 0–9 sind im [Arbeitsprotokoll](architecture-roadmap.md) mit
Änderungen, Prüfungen, Ergebnissen und offenen Punkten erfasst. Offen bleiben
die unabhängige Abschlussprüfung, menschliche Urteile und die empirische
Bestätigung musikalischer Verbesserungen; diese Ergebnisse wurden nicht erfunden.
