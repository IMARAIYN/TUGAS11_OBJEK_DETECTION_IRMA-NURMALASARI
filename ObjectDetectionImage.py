from ultralytics import YOLO
import os
import cv2

def detect_image(model, image_path, output_dir="output_images"):
    """
    Deteksi objek pada gambar menggunakan YOLO.
    :param model: Model YOLO yang sudah dilatih.
    :param image_path: Path ke gambar yang akan dideteksi.
    :param output_dir: Direktori untuk menyimpan hasil.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = model(image_path)
    for result in results:
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        annotated_image = result.plot()  # Gambar dengan anotasi deteksi
        cv2.imwrite(output_image_path, annotated_image)
    print(f"Hasil deteksi gambar disimpan di {output_dir}.")

# Load atau Train model
trained_model_path = "yolo11_trained.pt"
if os.path.exists(trained_model_path):
    print("Memuat model terlatih...")
    model = YOLO(trained_model_path)
else:
    print("Model tidak ditemukan. Melatih model baru...")
    model = YOLO("yolo11n.pt")
    train_results = model.train(
        data=r"C:\Users\62857\Pictures\opencv\ipyn\object_detection\coco\coco8.yaml",
        epochs=100,
        imgsz=640,
        device="cpu",
    )
    model.save(trained_model_path)

# Path gambar
image_path = r"C:\Users\62857\Pictures\opencv\ipyn\object_detection\image\person2.jpg"

# Deteksi objek pada gambar
detect_image(model, image_path)
