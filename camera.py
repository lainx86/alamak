from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
	model_path = Path("best.pt")
	if not model_path.exists():
		print(f"Model tidak ditemukan: {model_path.resolve()}")
		return

	model = YOLO(str(model_path))

	camera_index = 2
	cap = cv2.VideoCapture(camera_index)

	if not cap.isOpened():
		print(f"Gagal membuka kamera pada index {camera_index}.")
		print("Pastikan kamera terhubung dan index benar.")
		return

	print("Model berhasil dimuat. Tekan 'q' untuk keluar.")

	while True:
		ret, frame = cap.read()
		if not ret:
			print("Gagal membaca frame dari kamera.")
			break

		results = model.predict(source=frame, conf=0.25, verbose=False)
		annotated_frame = results[0].plot()

		cv2.imshow("YOLO Detection (best.pt)", annotated_frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
