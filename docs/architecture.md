# Architektur: Phrase Mapper

Stand: 28.10.2026. Aktueller Produktionsvertrag: `phrase-mapper-v5`, Modellformat 5,
Ereignisschema `cubic-phrase-plan-v3`, Targetschema `musical-plan-targets-v3`,
Vorbereitungsmanifest 3. Rhythmus, Plan, Objektdecoder und Kontextbewerter gehören
zu **einem gemeinsam trainierten Safetensors-Artefakt**. Deterministische Auswahl,
Guards und `.osu`-Writer bleiben eigenständige Teile des Hybridansatzes.

Modelle, Optimiererzustände und Targets früherer Formate werden nicht konvertiert.
Neue Gewichte erfordern neue Aufbereitung und vollständiges Training; vorhandene
Dateien werden nicht gelöscht. Historische v3-Messungen in den Benchmarkberichten
beschreiben eingefrorene Vergleichsstände, keinen zweiten Produktionspfad.

[Plan und Entscheidungen](architecture-roadmap.md),
[Zeitrepräsentation und Plan](musical-time.md) und
[Vergleichsexperimente und Evidenzgrenzen](architecture-evaluation.md) ergänzen
diese Beschreibung des aktuellen Codes. Technische Korrektheit, Referenzähnlichkeit
und musikalische Qualität werden getrennt bewertet.

## Audio und musikalischer Kontext

FFmpeg dekodiert zu 22.050 Hz Mono. 64 Log-Mel-Bänder, FFT 2048 und Hop 256 liefern
etwa 11,61 ms je Frame. Ein absoluter RMS-Boden erhält digitale Stille trotz
relativer Lautstärkennormalisierung. Sehr leise hörbare Ausklänge unter diesem
Boden können dennoch verloren gehen. Die frühere zusätzliche relative
Lautstärkeschwelle der Onsetauswahl wurde entfernt. Trainingsaudio und Mapinhalt
werden für die Aufbereitung gehasht.

Pro Beispiel: 2.048 Frames (23,78 Sekunden), davon 1.024 zentrale Frames als
Rhythmusziele. Zwei gelernte Faltungen verdichten auf 512 Audiotokens. Für den
ganzen Song entstehen 128 zeitlich geordnete Summarytokens aus Mel-Mittelwert und
-Maximum. Ein eigener Transformer verarbeitet diesen Überblick; die detaillierten
Audiotokens können anschließend auf die globale Erinnerung zugreifen. Der Encoder
behält Sinus-/Kosinusmerkmale mit Sekundenperioden von 1/8 bis 16 sowie relative
Songposition. Das allein behauptet keine Beatphase oder Takteins.

| Größe | Breite / Heads | Audio- / Objektdecoder-Blöcke | Parameter |
|---|---|---|---:|
| Kompakt | 128 / 4 | 4 / 3 | 2.286.838 |
| Standard | 256 / 8 | 6 / 4 | 11.322.486 |
| Groß | 384 / 8 | 8 / 6 | 33.451.382 |

Alle drei Größen haben zwei globale Blöcke. Größe ist keine Qualitätsrangliste;
mehr Kapazität braucht geeignete Daten und Rechenbudget.

Der hochauflösende Rhythmuszweig kombiniert Transformermerkmale mit lokalen
Audiodetails und sagt Onsets, Beatpulse, lokale Notendichte und logarithmische
Beatlänge voraus. AR, CS, OD, HP, Dichte, typische Bewegungsweite,
Richtungskontinuität und Slideranteil sind acht globale Conditions. Die letzten
drei werden aus Trainingsmaps gemessen; sie sind keine gesicherten Stilklassen.
OD ist Trefferfensterstrenge und kein Star-Rating.

## Explizite musikalische Zeit

`TimingPoint` erhält Zeit, Beatlänge, vererbte Geschwindigkeit, Meter,
Beat-/Metrikkonfidenz und optionalen Taktursprung. Importierte rote Timingpoints
liefern Referenzangaben. Geschätzte Pulse oder manuelle BPM/Offset allein liefern
keine Takteins; deren Metrik bleibt unbekannt. Triolen belegen keinen 3/4-Takt,
regelmäßige Pulse lösen Halb-/Doppeltempo nicht automatisch.

Die Objektquery enthält nun 31 statt 20 Merkmale: die bisherigen Sekunden- und
Nachbarkopfmerkmale plus elf gemeinsame Train-/Inferenzmerkmale für integrierte
Beatposition, lokale Beatphase, Beatlänge und Konfidenz, Taktphase/Meter und
Konfidenz, timingabgeleitete Downbeatnähe und Timingwechselnähe. Downbeatnähe ist
keine gelernte Audiowahrscheinlichkeit. Fehlende Metrik wird maskiert; vor dem
ersten belegten Timingpoint sind die musikalischen Merkmale unbekannt.

Trainingsaugmentation lässt Timing oder Metrik weg und stört teilweise Phase und
Tempo gemeinsam. Damit sieht der Decoder nicht ausschließlich perfekte
Referenzangaben. Die Referenzobjekte bleiben unverändert. Der interne
Ablationsschalter `enable_musical_time` blendet nur die elf neuen Kanäle aus;
Parameterkapazität und übriger Modellpfad bleiben gleich.

Vollständige manuelle Vorgaben sind über `GenerateRequest.timing_points` möglich:
höchstens 256 geordnete Abschnitte mit Zeit und Beatlänge in ms sowie optional
Meter und Taktursprung. Ein angegebener Taktursprung muss zum roten Timingpoint
modulo ganzer Takte passen, damit der Export dieselbe Metrik darstellen kann.
Bekannter Meter bleibt beim Export erhalten. Bei unbekanntem Meter schreibt
`.osu` syntaktisch 4; Bericht und Vorschau halten die Ungewissheit fest, und der
Dateidefault beeinflusst die Generierung nicht. Vor dem ersten positiven
manuellen Timingpoint verwenden Slider den technischen Parserdefault von
500 ms pro Beat mit Konfidenz null; so stimmen Dauer und SV im Roundtrip überein.

## Kompakter Mappingplan

Ein kleiner Planhead auf den 128 vorhandenen Songtokens lernt sieben kombinierbare
Proxys: Aktivität, unbedeckter Zeitanteil, Comboanfangsrate, Slideranteil,
Sprungweite, Richtungskontinuität und Regelmäßigkeit kurzer Intervallfolgen.
Der **vorhergesagte** Plan wird in die globale Erinnerung projiziert und kann
Rhythmus und Objektdecoder beeinflussen. Referenzplanlabels sind keine Eingaben.
Jede Summarybin wird pro Songdurchlauf einmal beaufsichtigt, auch wenn ein
Audiofenster mehrere Objektbatches benötigt.

Diese Analyseabschnitte sind keine erkannten musikalischen Phrasen. Der Plan
trennt globale Schwierigkeit von beschreibenden Rhythmus-/Bewegungsmerkmalen,
liefert aber keine verlässlichen Jumps-, Streams/Bursts-, Alt-, Tech-, Precision-
oder Reading-Labels. Mehrere Ausrichtungen können dieselben Proxys erklären und
gleichzeitig auftreten. Musikalischer Layer bleibt unbekannt; es werden keine
Instrumenten- oder Gesangslabels unterstellt. Comboanfänge sind nur ein schwacher
Akzent-/Gruppierungsproxy. Es gibt keinen pauschalen Glättungsverlust, der echte
Stilwechsel unterdrücken soll.

Der interne Schalter `enable_plan` ermöglicht einen Vergleich mit denselben
Parametern, aber ohne Planprojektion und Planhilfsverlust. Seine Wirkung muss mit
passenden Gewichten separat beurteilt werden. [Details und Masken](musical-time.md).

## Zusammenhängende Hitobject-Folgen

Der kausale Decoder nutzt bis zu 128 Ereignisse. Jede Eingabe enthält das vorherige
Objekt: acht Kurvenkoordinaten, tatsächlichen Endpunkt nach Wiederholungen, Dauer,
Typ, Repeats, Zeitabstand, Combo und Sprungweite. Die aktuelle Query kennt den
provisorischen Kopfzeitpunkt und vorige/nächste Zeitlücke; die nächste Position
ist niemals Eingabe. Cross-Attention verbindet Historie und Audio. Im Training
werden höchstens 64 neue Ziele pro Beispiel mit älteren Objekten als Kontext
bewertet. Rhythmus- und Planziele werden bei mehreren Objektteilen nicht mehrfach
gewichtet.

Acht gemeinsame Mischkomponenten beschreiben Kopf und drei weitere Punkte einer
kubischen Bézier-Kurve, jeweils mit acht Mittelwerten/Skalierungen und einem
Komponentengewicht. Mehrere gültige Platzierungen können nebeneinander bestehen,
statt zwingend zu einer mittleren Kurve zu verschmelzen. Modenkollaps ist damit
nicht ausgeschlossen. Weitere Ausgänge sind Circle/Slider, 19 Beatdauern von
1/12 bis 16 inklusive Triolen, 1–4 Durchläufe, New Combo und acht Bewegungsgrößen:
Sprungweite, freie Geschwindigkeit, Winkelsinus/-kosinus, Geschwindigkeitsänderung,
Sliderreiseweg/-geschwindigkeit und Verhältnis aufeinanderfolgender Sprungweiten.

Horizontale/vertikale Spiegelung und Audiogain bleiben Trainingsaugmentationen.
Unabhängige Störungen von Vorgängerkontrollen und Endpunkten wurden entfernt, weil
sie Kurve und abhängige Ziele inkohärent ändern konnten. Das Training bleibt
überwiegend Teacher Forcing. Freie Generierung verwendet die tatsächlich gewählten
und reparierten Objekte einschließlich Endpunkt und Repeat-Parität.

## Hauptziele, Hilfsverluste und Trainingssteuerung

| Hauptziel | Bewertung | Gewicht |
|---|---|---:|
| Onsets | Gewichtete binäre Kreuzentropie, benachbarte Frames als weiche Ziele | 1 |
| Beatpulse | Binäre Kreuzentropie mit Timingmaske | 0,3 |
| Lokale Dichte | Robuster Fehler der weiterhin über vier Sekunden geglätteten Zielkurve | 0,5 |
| Tempo | Robuster Fehler der logarithmischen Beatlänge mit Timingmaske | 0,2 |
| Gemeinsame Geometrie | Maskierte negative Log-Likelihood der Mischverteilung | 1 |
| Objekttyp | Kategoriale Kreuzentropie | 0,5 |
| Sliderdauer | Kategoriale Kreuzentropie auf unterstützten Slidern | 0,3 |
| Repeats | Kategoriale Kreuzentropie auf unterstützten Slidern | 0,2 |
| New Combo | Binäre Kreuzentropie | 0,1 |
| Bewegung | Robuster Fehler der acht Bewegungswerte | 0,3 |
| Bewegungsfolgen | Abgeleitete Kandidatenbewegung gegen Referenz | 0,15 |
| Kontextverträglichkeit | Schwache Präferenz für Referenzstil gegenüber passenden Kontrafaktualen | 0,1 |

Das Viersekunden-Dichtetarget ist eine vorhandene Trainingshüllkurve; die Inferenz
verwendet keine festen Viersekundenbudgets mehr. Planverlust mit Gewicht 0,1 und
der optionale Eigenkontextverlust werden getrennt protokolliert und nicht in
vergleichbare Hauptkurven oder Checkpointwahl eingerechnet. Der beste Checkpoint
bleibt der mit dem niedrigsten Validierungsgesamtwert der zwölf Hauptziele.
Niedrigerer Loss ist kein Beleg für bessere musikalische Qualität. Eine negative
kontinuierliche Geometrie-NLL ist zulässig.

Der Kontextbewerter verwendet strukturgleiche Donorobjekte und ganze Spiegelungen/
180°-Drehungen. Typ, Repeatstruktur, ähnliche Dauer, gültige Quell-/Zielslots und
symmetrische Boundsunterstützung werden geprüft. Bewegung wird aus der tatsächlichen
Referenzhistorie neu berechnet. Nahezu gleichwertige Alternativen liefern kein
Negativsignal; größere Abweichungen höchstens eine schwache Präferenz bis 0,75,
kein sicheres Falschlabel. Dies ist ein Kompatibilitätsbewerter ohne menschliche
Qualitätslabels. Die Präferenz für den Referenzstil bleibt eine überprüfbare Annahme.

Training verwendet AdamW, Weight Decay, Gradient Accumulation, Gradient-Clipping
auf Norm 1 und einen epochenbudgetunabhängigen Warmup von höchstens 200 Updates
bzw. einer Epoche. Drei Validierungen ohne Verbesserung um mindestens 0,0001
halbieren die Lernrate bis auf 5 % der Ausgangsrate; kürzere Frühstopp-Geduld
verkürzt dieses Intervall. Jede echte Verbesserung speichert ein Modell.
Standardmäßig endet Training nach zwölf Epochen ohne neuen Bestwert und gewährt
nach einer Reduktion mindestens ein Reduktionsintervall. Controllerzustand,
Lernrate und Zähler werden atomar gespeichert. Ohne Validierung erfolgen weder
Plateau-Reduktion noch Frühstopp.

Nicht-endliche Verluste, Gradienten und Gewichte stoppen Updates. Numerisch stabile
Operationen umgehen bekannte DirectML-Rückwärtsprobleme; Maxima zur Stabilisierung
werden vom Gradientengraphen getrennt. DirectML kann AdamW-`lerp` auf CPU ausführen.
Es ist keine CUDA-spezifische Attention-Erweiterung erforderlich.

### Experimenteller eigener Positionskontext

Die Option bleibt **standardmäßig aus**. Ab Epoche 5 steigt die Wahrscheinlichkeit
geeigneter Vorgänger über zehn Epochen von 2 auf maximal 20 %. Abgetrennte
Vorhersagen des vorigen Objekts verschieben Kopf, Kontrollen und paritätsabhängigen
Endpunkt gemeinsam um höchstens 16 Pixel und gemeinsam innerhalb der Grenzen.
BOS/Padding bleiben erhalten, Abstandsindikatoren werden nachgeführt. Zeiten,
Typen und Dauern der Vorgeschichte bleiben referenzbasiert. Das ist kein frei
ausgerolltes Training vollständiger Mapfolgen.

Der zusätzliche Decoderlauf lernt unter veränderter Historie ausschließlich Typ,
Dauer, Repeats und Combo mit Gesamtgewicht 0,1. Er erzwingt keine ursprüngliche
absolute nächste Referenzgeometrie. Geometrie-, Bewegungs- und Kritikerziele bleiben
auf dem kohärenten Referenzpfad; der zusätzliche Audiokontext ist abgetrennt.
Indirekte Folgen späterer Optimierungsschritte bleiben möglich. Der CUDA-Graph
wird beim Aktivieren einmal neu aufgezeichnet und liest anschließend die
Wahrscheinlichkeit als Tensor. Die historischen v3-Zusatztrainingsresultate vom
27. Oktober betreffen einen anderen Geometriehilfsverlust und validieren diese
neue Variante nicht; siehe [Trainingsvergleiche](training-validation.md).

## CUDA-Ausführung

CUDA verwendet PyTorch SDPA mit denselben Padding-/Kausalmasken; CPU und DirectML behalten die explizite Attention. Automatische Rechengenauigkeit nutzt auf geeigneter CUDA-Hardware BF16 nur im Netzvorwärtslauf. Die Ausgänge werden vor der Verlustberechnung nach Float32 zurückgewandelt; insbesondere Mischverteilungen, Bewegungsableitungen und Kontextbewertung bleiben Float32. Mastergewichte, AdamW-Zustand, Validierung und Modell-Dateien bleiben Float32. TF32 ist deaktiviert. „Float32“ in der UI ermöglicht einen Vergleich ohne BF16.

CUDA-AdamW nutzt den integrierten Fused-Pfad. Ein Hintergrundthread bereitet maximal einen Folgebatch in gepinntem RAM vor; Transfers erfolgen nicht blockierend. Bewegungsziele werden vektorisiert berechnet. Verluststatistiken sammeln sich abgetrennt vom Gradientengraphen auf der GPU, statt jeden Teilverlust einzeln zur CPU zu übertragen. Endlichkeitsprüfung und Gradient-Clipping bleiben vor jedem Update erhalten; Fortschrittsmeldungen werden auf ungefähr einmal pro Sekunde begrenzt.

Bei aktivierten CUDA Graphs und genau einem Batch pro Update werden Vorwärtslauf, Verluste und Rückwärtslauf einmal aufgezeichnet. Wiederholungen lesen neue Daten und aktuelle Gewichte, verwenden neue Dropout-Masken und überschreiben die Gradienten. Optimierer, Lernrate, Clipping, Abbruch und Checkpoints bleiben außerhalb des Graphs. Abweichend große letzte Batches laufen normal; Gradient Accumulation verwendet ebenfalls den normalen CUDA-Pfad. Der Graph hält feste Puffer und einen privaten Speicherpool, braucht daher mehr reservierten Speicher als ein Lauf ohne Graph. Vorbereitungsdurchläufe ändern keine Gewichte; die Zufallszustände werden danach wiederhergestellt.

Pro Epoche: gemessene Trainingsdauer, Beispiele/s, Updates, Genauigkeit und CUDA-Spitzenwerte für belegten/reservierten Speicher. Erste Epoche inklusive Graph-Vorbereitung, ohne vorherige Audioaufbereitung und nachfolgende Validierung; innerhalb der Schleife geschriebene Checkpoints zählen mit. Das sind keine GPU-Gesamtspeicherwerte: Treiber, Desktop und andere Programme kommen hinzu. [Historische CUDA-Messungen und Grenzen des Qualitätsnachweises](cuda-performance.md); diese Durchsatzwerte gehören zum dort dokumentierten Modellstand, nicht automatisch zu v5.

## Datenaufbereitung und unabhängige Prüfung

Audiohash und normalisierte Künstler-/Titelmetadaten verbinden Maps transitiv zu
stabilen Songgruppen. Alle Schwierigkeiten einer Gruppe bleiben vollständig im
Training, in der Validierung (~15 %) oder im Test (~10 %, ab fünf Gruppen).
Der Split hängt am Seed. Mehr Schwierigkeiten eines Songs ersetzen keine
musikalische Vielfalt. Die Heuristik ist kein akustischer Dublettennachweis.

Der Parser berücksichtigt Meter, SliderMultiplier, vererbte Geschwindigkeit,
deklarierte Pfadlänge und Repeat-Parität. Slider werden auf eine kubische Bézier
angepasst. Unterstützt sind höchstens acht Pixel Anpassungsfehler, vier Durchläufe
und 16 Beats. Nicht passende Slider tragen weiter Kopfposition und Typ bei,
jedoch keine falschen Kurven-/Dauerziele. Spinnerzeiten werden maskiert. Ungültige
Koordinaten, Objekte außerhalb des Audios und Kollisionen unter Frameauflösung
schließen Maps mit protokolliertem Grund aus. Die versionierten Caches enthalten
Events, Framelabels, Timing und Planwerte/-masken mit verifizierten Inhaltshashes.

Teacher-Forcing-Validierung aggregiert nach gültigen Frames/Objekten und meldet
Teilverluste, Onset-Präzision/Recall/F1 mit Eins-zu-eins-Matching ±35 ms,
Typgenauigkeit und Bewegungsfehler. Die UI trennt Rhythmus, Geometrie, Struktur
und Bewegung/Zusammenhang; Gesamtverlust, Lernratenwechsel, beste Epoche und
Songaufteilung bleiben sichtbar. Plan- und Kontextwerte bleiben Hilfsdiagnostik.

Freie Validierung läuft optional in Epoche 1, alle fünf Epochen und am Ende auf
bis zu drei festen verschiedenen Songs. Pro Song werden bis zu sechs feste
24-Sekunden-Proben für Anfang/Mitte/Ende, geringe/hohe Audioaktivität und einen
Fensterübergang ausgewählt; doppelte Bereiche werden zusammengeführt. Aktivität
ist keine erkannte Refrain- oder Breakannotation. Jeder Abschnitt startet mit
eigener generierter Historie, verwendet aber die vollständige Audioübersicht.
Maximal 128 Objekte pro Abschnitt; Begrenzungen und Fehler werden berichtet.
Der Referenzhorizont wird vorher festgelegt und nicht durch das letzte erzeugte
Objekt verkürzt. Die echte nächste Kopfzeit außerhalb des Ausschnitts bleibt für
Queries und Sliderplanung verfügbar.

Nach Laden der besten Gewichte wird zusätzlich höchstens ein vollständiger Song
mit maximal 300 Sekunden und 2048 Objekten geprüft. Abgeschnittene Läufe gelten
nicht als vollständig. Die Abschlussevaluierung nutzt die separaten Testsongs;
fehlen diese, wird die Verwendung von Validierung ausdrücklich markiert. Diese
Läufe ändern keine Lernrate oder Checkpointwahl. Berichte enthalten Zeit-/Endpunkt-
Roundtripfehler, Reparaturen, Bewegungsquantile, Referenzähnlichkeit und Laufzeit.
Abschnittsneustarts messen keine über Minuten akkumulierten Kontextfehler.

Das Korpusinventar berücksichtigt frühere Trainings-/Validierungs-/Testexposition,
Dublettenhinweise und Maskierungen. Im untersuchten vorhandenen Korpus ließ sich
kein frischer unabhängiger Abschlusstestsatz zurückhalten: alle Gruppen waren
bereits exponiert. Ein früherer Testsplit wird deshalb nicht nachträglich als
unberührter Architekturtest ausgegeben. Neue Songs und verblindete menschliche
Playtests sind weiter erforderlich. Aktuelle Zahlen und Einschränkungen stehen
im [Evaluationsbericht](architecture-evaluation.md).

## Generierung: Zuständigkeiten und begrenzte Suche

1. **Modell:** Audiokontext und Songplan verarbeiten; Rhythmus-/Tempoausgaben und
   bedingte Objekt-/Geometrie-/Bewegungsverteilungen liefern. Variation bleibt
   über mehrere gelernte Modi und zusammenhängende Kurventransformationen möglich.
   Es werden keine zufälligen Timingfehler oder unabhängigen Kontrollpunktstörungen
   ergänzt.
2. **Timing:** Konsistente gelernte Pulse über mehrere Beats zu Tempoabschnitten
   fitten, um Framequantisierungsdrift zu vermeiden. Unsicheres Timing fällt auf
   eine konstante Audioschätzung zurück. Manuelle konstante oder vollständige
   Timingvorgaben können dies ersetzen; Metrikunsicherheit bleibt ausdrücklich.
3. **Rhythmusauswahl:** Ein deterministischer Beam der Breite 8 vergleicht Auswahl
   und Auslassen von Onsetkandidaten. Dichte wirkt über kontinuierlich abklingende
   Belegung als weiches Ziel. Schwache Peaks werden nicht zum Auffüllen erzwungen;
   starke isolierte Akzente bleiben auch bei sehr niedrigem Dichtetarget möglich.
   Kurze geordnete Intervallwiederholungen und Rhythmusfamilien wirken begrenzt
   auf den Score. Längere Pausen beenden lokale Kontinuität. Greedy ist eine
   Benchmarkablation, kein Legacy-Modellpfad.
4. **Lokales Snapping:** Nur bei ausreichender Beatkonfidenz und konsistenten
   Abständen/Phasen werden passende gerade und triolische Unterteilungen verglichen.
   Mehrdeutigkeit innerhalb einer halben Audioframeauflösung bevorzugt die gröbere
   passende Unterteilung. Kein pauschales gemeinsames Raster und keine erfundene
   Takteins. Post-Snap-Kollisionen unter 35 ms werden nach Evidenz aufgelöst;
   Stille wird vor und nach dem Snapping geprüft.
5. **Gemeinsame Zeit-/Objektplanung:** Circle und Slider, maximal zwei passende
   Dauerklassen und zwei Repeatzahlen werden zusammen mit Geometriekandidaten
   bewertet. Das Budget beträgt standardmäßig zwölf Kandidaten je Zeitplan.
   Neben dem regulären Plan darf ein zweiter Plan genau einen deutlich schwächeren
   Innenkopf innerhalb von zwei Beats bei überwiegend hörbarer, spektral stetiger
   Textur auslassen. Der gewählte Slider muss ihn tatsächlich überdecken. Das ist
   ein konservativer Sustainproxy, keine Instrumentenannotation. Starke unabhängige
   Akzente dürfen diesen Weg nicht eröffnen. Der Decoder sieht bereits die nächste
   Kopfzeit des jeweiligen Plans; erst die Wahl verändert die endgültigen Zeiten.
6. **Kandidatenbewertung:** Geometrie-Log-Likelihood pro genutzter Dimension,
   Typwahrscheinlichkeit, entropiezentrierte Dauer-/Repeatplausibilität, relative
   Bewegung, Reparaturen, Kopfüberdeckung und Kontextverträglichkeit werden getrennt
   gewichtet. Ganze Kurven werden gemeinsam verschoben, gedreht oder skaliert.
   Kein einmaliger zufälliger Typentscheid für alle Geometrien. Kandidaten tragen
   ihre endgültige Dauer, Form und ihren paritätsabhängigen Endpunkt weiter.
7. **Optionale Fortsetzungen/Motive:** `lookahead` bleibt standardmäßig aus und
   vergleicht höchstens drei kurze Zweige mit genau einer nächsten Aktion. Jeder
   führt eigene endgültige Historie und identisch geklonte Proposal-RNG-Zustände;
   Scores werden über betrachtete Aktionen gemittelt. `motifs` bleibt ebenfalls
   aus: spektrale Sequenzähnlichkeit verknüpft höchstens 512 Analyseabschnitte von
   acht Beats innerhalb desselben Songs. Das sind keine behaupteten Takte oder
   sicheren musikalischen Phrasen. Nur bei passender bereits gewählter Rhythmik
   werden relative Bewegungen weich verglichen und höchstens eine ganze
   transformierte Geometrie innerhalb des bestehenden Budgets vorgeschlagen.
   Keine Koordinaten fremder Songs, keine Gleichsetzung von Energie mit Motiv,
   keine starre Wiederholung.
8. **Guards und Export:** Gültige Kandidaten auswählen, konservative Policies
   anwenden und bei ungeeigneten Slidern einen protokollierten Circlefallback
   erzeugen. Der deterministische Writer schreibt kubische Kurven, Repeats,
   New Combos, rote Timingpoints mit Meter und passende grüne SV-Punkte.
   Anschließendes Parsen kontrolliert Kopfzeiten, Anzahl, Typen, Repeats, Dauer,
   Endpunkte und zeitliche Überschneidungen. `.osz` enthält die Map und Ogg-Audio;
   das Modell lernt keine Dateisyntax.

Ein optionaler Trace protokolliert Onsetrohzeit, Evidenz und Gründe, weiche
Dichteziele, Snapverschiebungen/Kollisionen, vorgeschlagene und endgültige
Objekte, Reparaturen/Fallbacks sowie Scorebestandteile und Fortsetzungszweige.
Er verändert keine RNG-Zustände oder Generierungsentscheidungen.

## Technische Guards, unterstützter Stil und Grenzen

Endlichkeit, Dateisyntax, konsistente Endpunkte und gültige SV sind technische
Anforderungen. CS-Innenränder, mindestens 35 ms zwischen Sliderende und nächstem
Kopf sowie eine feste Notfallbewegungsgrenze von 1500 Pixeln/s (maximal 600 Pixel)
sind konservative Generatorpolicies. OD skaliert diese Bewegungsgrenze nicht.
Partielle Kopfüberdeckung, Bewegungsabweichung und Motiventfernung sind weiche
Präferenzen. Langsamere Bewegung allein bedeutet keine bessere Map.

Die Kurvenprüfung nutzt 65 Punkte und verwirft Selbstkreuzungen, scharfe Haken,
starke Umwege und doppelte benachbarte Kontrollen. Letztere trennen im `.osu`-
Bézierformat Segmente. Normale Bögen und S-Kurven bleiben möglich; die Prüfung ist
kein mathematischer Vollständigkeitsbeweis und keine offizielle Rankingregel.
Absichtliche Tech-/Reading-Spezialmuster können durch diese unterstützte
Stilgrenze weiterhin ausgeschlossen werden. Der Bericht nennt Formprobleme und
Umwegquantile; weniger Guardverletzungen belegen allein keine Musikqualität.

Es gibt kein offizielles Star-Rating oder Ranked-Qualitätsversprechen. Sliderpfad-
Interaktionen, genaue Stackingsemantik, langfristige Ermüdung, freie Kontextfehler
und semantische musikalische Layer sind nicht vollständig modelliert. Hitsounds,
Storyboard, Kiai, Spinner und SliderTickRate werden nicht gelernt; exportierte
TickRate ist 1. Die Webvorschau zeigt Kurven, Repeats, AR und CS sowie Previewklänge,
ist aber kein identischer osu!-Renderer. Menschliche A/B-Bewertungen bleiben nötig.

## Artefakte, Caches und Wiederaufnahme

`model.safetensors` enthält den besten gemeinsamen Inferenzstand;
`training.safetensors` den letzten Optimierer-/Gewichtsstand mit Schritt/Epoche
und expliziten Format-/Schemaangaben. Checkpoints entstehen nach dem ersten
Update, alle 500 Updates und an Epochengrenzen; Dateien werden atomar ersetzt.
Ein Abbruch nach mindestens einem Update erhält den Lauf. Nach hartem Abbruch
kann der Appstart kompatible `recovery.json`-Stände registrieren.

Resume verlangt aktuellen Artefaktvertrag und denselben unveränderten Datensatz.
Modellkonfiguration und Split werden übernommen; Controllerzustand wird erhalten.
Ein unterbrochener Epochenteil kann erneut gesehen werden, die Zufallsfolge wird
nicht bitgenau fortgesetzt. Neue Trainingssongs erfordern ein neues Modell.
Download-ZIPs enthalten nur Inferenzgewichte und JSON, keinen Optimierer. Eine
vollständige Sicherung benötigt die lokalen Daten. Inkompatible Modelle und
Trainingscheckpoints werden verständlich abgewiesen; es gibt keine Migration
oder Gewichtskonverter.

Audio-/Target-/Plancaches sind atomare NumPy-Dateien ohne Pickle, mit Memory-Mapping
und verifizierten Manifesten. Formatänderungen verwenden neue Namen und bauen
Targets neu auf. Große Sammlungen benötigen weiterhin Aufbereitungszeit und
Cacheplatz. FastAPI bedient React, ein Worker verarbeitet Jobs seriell, SQLite
speichert Status und portable `data://`-Pfade. Modellimport akzeptiert nur den
aktuellen Architektur-/Format-/Targetvertrag, geprüfte Konfiguration, endliche
Float32-Safetensors und JSON; er führt keine heruntergeladenen Programme aus.

## Fachliche Referenzen

- [osu!-Dateiformat](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29):
  technischer Vertrag für Timing, Meter, Slider und Export.
- [Timing](https://osu.ppy.sh/wiki/en/Beatmapping/Timing),
  [Ranking Criteria](https://osu.ppy.sh/wiki/en/Ranking_criteria) und
  [osu!standard-Kriterien](https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!):
  technische Regeln und Empfehlungen von eigenen Generatorpolicies trennen.
- [Beatmap-Tags](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags),
  [musikalische Layer](https://osu.ppy.sh/wiki/en/Music_theory/Layer) und
  [Time-distance equality](https://osu.ppy.sh/wiki/en/Beatmapping/Mapping_techniques/Time-distance_equality):
  Stil-/Interpretationshilfen, keine vollständigen Trainingslabels oder Pflicht
  zur allgemeinen Glättung.
- [osu! Difficulty Preprocessing](https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Osu/Difficulty/Preprocessing/OsuDifficultyHitObject.cs):
  fachlicher Bezug für Bewegungsgrößen, keine übernommenen Star-/Skillformeln.
- [Dance Dance Convolution](https://proceedings.mlr.press/v70/donahue17a.html) und
  [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator): methodische
  Vergleiche; keine übernommenen Gewichte oder behauptete Ergebnisgleichheit.
- [DirectML-Operatorübersicht](https://github.com/microsoft/DirectML/wiki/PyTorch-DirectML-Operator-Roadmap):
  Plattformverträglichkeit wird zusätzlich auf echter Hardware geprüft.
