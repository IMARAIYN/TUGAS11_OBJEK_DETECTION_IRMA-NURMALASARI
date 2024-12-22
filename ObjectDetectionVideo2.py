import cv2
from ultralytics import YOLO
import os

def detect_video(model, video_path, output_dir="output_videos", output_name="output_video.mp4"):
    """
    Deteksi objek pada video menggunakan YOLO.
    :param model: Model YOLO yang sudah dilatih.
    :param video_path: Path ke video yang akan dideteksi.
    :param output_dir: Direktori untuk menyimpan hasil video.
    :param output_name: Nama file video output.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka video.")
        return

    # Mendapatkan ukuran frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Menggunakan codec 'XVID' untuk video output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Memproses video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Memprediksi dan memplot hasil pada frame
        results = model.predict(source=frame, imgsz=640, conf=0.25)
        for result in results:
            rendered_frame = result.plot()  # Render hasil deteksi pada frame
            out.write(rendered_frame)  # Menulis frame dengan anotasi ke video output

        # Menampilkan video secara langsung (opsional, tekan 'q' untuk keluar)
        cv2.imshow("YOLO Detection", rendered_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Hasil deteksi video disimpan di {output_video_path}.")

# Load model
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

# Path video
video_path = r"C:\Users\62857\Pictures\opencv\ipyn\object_detection\video\testvideo.mp4"

# Deteksi objek pada video
detect_video(model, video_path)
