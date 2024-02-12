from ultralytics import YOLO

# Carga el modelo
model = YOLO('/yolov8x.pt')
print("Modelo cargado")

result = model.predict(source='0', show=True) # predict and save
print("Predicción realizada")