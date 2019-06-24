# Applied-AI-Technologies

Verwendung dieser Dateien:

Tensorflow Object Detection nach diesem Tutorial installieren: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#exporting-a-trained-inference-graph

Zum Trainieren: Folge dem Tutorial mit den Bildern und dem Pre-trained-model aus Drive.

Datensatz und Pre-trained-Modell auf Google Drive. Für Verwendung in Ordner Workspace_TF_Object speichern.

https://drive.google.com/open?id=1x8Ie_WagbWyF9zqU09HSae3jIYmk7I5E

https://drive.google.com/open?id=1N6RUCqljY3HkYIF_nLoBoqExjp6xGCIq

Zum direkten Abspielen Dateien speichern und Object_detection_video.py verwenden. Oder hse-aai-2019-team4.ipynb abspielen.

Für Verwendung auf Raspi:

https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html#install-package

Verzeichnis in Home anlegen

Dateien aus Files_to_run_on_raspi in Verzeichnis speichern

python3 object_detection_demo_ssd_async.py \
    -m frozen_inference_graph.xml \
    -i cam  \
    -d MYRIAD \
    -pt 0.6

ausführen.

Genauere Details unter Vorgehensweise.docx