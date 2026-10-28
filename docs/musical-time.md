# Musikalische Zeit und kompakter Plan – 28. Oktober 2026

## Aufgabe 4: explizite musikalische Zeit

Hypothese: Der Objektdecoder kann mit ausdrücklicher Beatphase und Tempo besser
zwischen rhythmisch unterschiedlichen Kontexten unterscheiden als mit reinen
Sekundenperioden. Das ist eine prüfbare Qualitätsannahme; der nachgewiesene Verlust
des Timingpoint-Meters war dagegen ein technischer Darstellungsfehler.

Baseline: Stand vor Aufgabe 4, gleiche Daten, Seeds, Modellbreite und Trainingsbudget.
Der neue interne Schalter `enable_musical_time=False` setzt ausschließlich die elf
zusätzlichen Querykanäle auf null. Beide Versuchsarme haben dieselbe Parameterzahl
und benötigen neu trainierte Gewichte. Alte Gewichte werden nicht konvertiert.

`TimingPoint` erhält Zeit, Beatlänge, vererbte Geschwindigkeit, Meter, getrennte
Beat-/Metrikkonfidenz und optionalen Taktursprung. Referenzmaps liefern explizite
Timingangaben, keine Garantie auf musikalisch richtiges Timing. Neue Audios mit
geschätzten Beatpulsen und manuelle BPM/Offset ohne Taktursprung haben unbekannte
Metrik. Halb-/Doppeltempo und Takteins werden dadurch nicht automatisch gelöst.

Der gemeinsame Train-/Inferenzpfad erzeugt elf Querymerkmale: integrierte
kontinuierliche Beatposition; Sinus/Kosinus der örtlichen Beatphase; logarithmische
Beatlänge; Beatkonfidenz; Sinus/Kosinus der Taktphase; Meter; Metrikkonfidenz;
timingabgeleitete Downbeatnähe und Nähe eines Timingwechsels. Die Downbeatnähe ist
keine gelernte Audiowahrscheinlichkeit. Unbekannte Metrik setzt ihre fünf Kanäle
auf null. Vor dem ersten belegten Timingpoint bleiben alle Zeitmerkmale unbekannt.
Für Sliderdauern vor einem ersten positiven manuellen Timingpoint gilt der
technische osu!-/Parserdefault von 500 ms pro Beat, mit Beatkonfidenz null und
unbekannter Metrik. Das verhindert abweichende Dauern im Export-Roundtrip und
liefert ausdrücklich keine Grundlage für Snapping oder Taktakzente.
Der Audioencoder behält seine Audiorepräsentation: die zusätzlichen Angaben
werden erst am geplanten Objektzeitpunkt benötigt.

Trainingsbeispiele lassen deterministisch mit ihrem Augmentierungsseed Timing
oder Metrik weg und stören teilweise Phase und Tempo gemeinsam. Die Referenzziele
bleiben unverändert. Unabhängiges Rauschen auf den zehn Vorgängerkoordinaten wurde
entfernt, weil es Kurve und tatsächlichen Endpunkt inkohärent ändern konnte;
Spiegelungen und Audiogain bleiben erhalten. Das ist zugleich eine technische
Vorbereitung für die getrennten Kontexttrainingsexperimente aus Aufgabe 8.

Vollständige manuelle Vorgaben sind über `GenerateRequest.timing_points` nutzbar:
höchstens 256 streng geordnete Abschnitte mit `time` und `beat_length` in ms sowie
optional `meter` und `bar_origin`. Ein Taktursprung muss modulo ganzer Takte zum
roten Timingpoint passen, damit der `.osu`-Export dieselbe Metrik ausdrücken kann.
Alternativ bleiben die vorhandenen konstanten BPM/Offset-Vorgaben nutzbar.

Bekannter Meter wird unverändert exportiert. Weil `.osu` einen positiven Meter
verlangt, schreibt der Writer bei unbekanntem Meter syntaktisch 4; Bericht und
Vorschau halten die Ungewissheit ausdrücklich fest. Dieser Dateiformat-Default
fließt nicht zurück in die Generierungsentscheidung. Triolen setzen niemals 3/4.

Neue Verträge: `phrase-mapper-v4`, Modellformat 4, `cubic-musical-time-v2`,
`musical-time-targets-v2` und Vorbereitungsmanifest 2. Ereignis- und Labelcaches
liegen unter neuen Namen. Alte Modelle/Optimizer werden verständlich abgewiesen;
alte Dateien bleiben liegen. Caches werden neu vorbereitet.

Prüfkriterien: Meterwechsel und SV-/Repeat-Roundtrip; keine erfundene Metrik;
identische Querymerkmale bei gleichem Train-/Inferenzkontext; kontinuierlicher
Beatfortschritt über Tempowechsel; kontrollierte Maskierung und reproduzierbare
Störungen; endliche Gradienten; Ablehnung inkompatibler Artefakte. Freie
Generierung und menschliche musikalische Qualität werden getrennt bewertet.

Der isolierte C-Vergleich mit neuen Gewichten ist durchgeführt und im
[Ergebnisbericht](architecture-evaluation.md) ausgewertet. Musikalische Zeit
verbessert die kurze Probe weder beim Hauptverlust noch bei freier Referenz-F1.
Ein Gewinn musikalischer Qualität bleibt offen und benötigt menschliche
Bewertungen. Die Konfidenzen sind verfügbare Evidenzindikatoren, keine
kalibrierten Wahrscheinlichkeiten.

Ergebnis der technischen Prüfung von Aufgabe 4 vor der Planerweiterung:
49 gezielte Tests bestanden; vier Cache-/API-Tests separat bestanden. Der
TestClient benötigte für lokale Socketpaare eine Ausführung außerhalb der
Sandbox. Der blockierte Lauf wurde beendet. Keine musikalische Verbesserung
wurde aus diesen Tests abgeleitet.

## Aufgabe 5: kleiner durchgängiger Mappingplan

Hypothese: Lokale, kombinierbare Beschreibungen der Referenzinterpretation helfen
dem vorhandenen Songkontext, Aktivität, Pausen und Bewegungscharakter über mehrere
Objekte konsistenter zu organisieren. Ein neuer Transformer oder ein fremdes
Instrumentenmodell ist dafür zunächst nicht nötig.

Der Planhead arbeitet auf den vorhandenen 128 Songsummary-Tokens. Sie bleiben
Analyseabschnitte; ihre Grenzen sind keine behaupteten musikalischen Phrasen.
Sieben kontinuierliche Proxys werden gelernt: lokale Aktivität, unbedeckter
Zeitanteil, Comboanfangsrate, Slideranteil, Sprungweite, Richtungskontinuität
und Regelmäßigkeit kurzer Intervallfolgen. Die Projektion des **vorhergesagten**
Plans konditioniert die Songerinnerung, damit Rhythmus und Objektdecoder ihn
nutzen können. Referenzplanlabels gehen niemals als Modelleingabe ein.

AR/CS/OD/HP bleiben Schwierigkeitseigenschaften. Sprungweite und Richtung sind
visuelle Bewegungsbeschreibungen; kurze regelmäßige Folgen sind ein schwacher
Stream-/Bursthinweis; Slider und Combos beschreiben die gewählte Interpretation.
Hohe Werte dürfen kombiniert auftreten. Sie sind keine semantisch gesicherten
Jumps-/Streams-/Alt-/Tech-/Precision-/Reading-Labels: dieselben Messwerte können
mehrere dieser Ausrichtungen beschreiben. Insbesondere Precision hängt zusätzlich
von CS und Platzierung ab, Reading von visueller Interaktion und Timing. Diese
mehrdeutigen Labels werden weder aus Difficulty-Namen noch aus Ranked-Status
erfunden. Musikalischer Layer bleibt unbekannt; Gesang/Instrumente benötigen
Annotationen oder separat begründete Verfahren. Comboanfänge sind ebenfalls
kein verlässliches Akzentlabel.

Pausenlabels beschreiben Referenzmapping, nicht digitale Stille. Ein Circle belegt
für diesen Proxy 100 ms, ein Slider seine tatsächliche Dauer. Unsupported-Kurven
liefern keine Bewegungsproxyziele; Spinnerabschnitte werden maskiert. Jede
Summarybin zählt genau einmal pro Songdurchlauf, auch wenn ein Audiofenster
mehrere Objektbatches benötigt. Es gibt keinen allgemeinen Glättungsverlust:
begründete abrupte Stilwechsel bleiben möglich, musikalische Grenzen sind
weiterhin eine offene Analyseaufgabe.

Isolierter Vergleich: `enable_plan=False/True` bei identischer Parameterkapazität,
gleichen Songgruppen, Seeds und Trainingsbudgets. Ohne Plan sind Projektion und
Zusatzverlust abgeschaltet. `plan` bleibt ein separat protokollierter
Hilfsverlust; Checkpointwahl und vergleichbare Hauptkurven verwenden weiterhin
die zwölf Hauptziele. Der Kontextbewerter bleibt ein Kompatibilitätsbewerter.

Mit den zusätzlichen Plangewichten wird der gesamte gemeinsame Artefaktvertrag
erneut eindeutig angehoben: `phrase-mapper-v5`, Format 5,
`cubic-phrase-plan-v3`, `musical-plan-targets-v3`, Cachemanifest 3. Die alte
Architektur wird nicht parallel betrieben. Aufgabe-4-Artefakte mit Format 4
werden ebenfalls neu trainiert. Beide Ablationsschalter sind interne
Versuchseinstellungen und keine zusätzlichen Studio-Tuningregler.

Technische Prüfkriterien: gleiche Parameteranzahl im Planvergleich; Planlabels
beeinflussen den Vorwärtslauf nicht; Planvorhersagen beeinflussen Rhythmus und
Geometrie; Masken für unbekanntes Timing, Pausen und ungeeignete Geometrie;
keine mehrfache Zielgewichtung; endliche Plan-/Modellgradienten; Hilfsverlust
ändert die Auswahlmetrik nicht. Der durchgeführte C-Vergleich zeigt einen höheren
Teacher-Forcing-Verlust und eine höhere freie Referenz-F1 für Plan gegenüber
Zeit allein. Dieses gemischte Ergebnis ist kein Musikqualitätsbeleg; menschliche
A/B-Urteile bleiben erforderlich.

Ergebnis der technischen Prüfung nach Aufgabe 5: 59 gezielte Tests bestanden
(35,06 s), darunter die fünf separaten Planprüfungen, Zeitrepräsentation,
Cache-/Datenpfad, endliche Modellgradienten, gemeinsame Objektplanung,
Trace-Identität, Modellimport und Trainingsfortsetzung. `git diff --check` sauber.
Parameterzahlen mit beiden neuen Komponenten: Compact 2.286.838, Standard
11.322.486, Large 33.451.382; beide internen Ablationsschalter lassen die jeweilige
Kapazität unverändert. Dies ist ein Funktionsnachweis, kein Musikqualitätsurteil.
