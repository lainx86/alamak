from pathlib import Path

import cv2
from ultralytics import YOLO


def main() -> None:
	model_path = Path("best.pt")
	if not model_path.exists():
		print(f"Model tidak ditemukan: {model_path.resolve()}")
		return

	model = YOLO(str(model_path))

	print("Daftar class pada model:")
	print(model.names)

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

		height, width, _ = frame.shape
		camera_center_x = width // 2
		camera_center_y = height // 2

		results = model.predict(source=frame, conf=0.25, verbose=False)
		result = results[0]

		annotated_frame = result.plot()

		red_buoys = []
		green_buoys = []

		for box in result.boxes:
			cls_id = int(box.cls[0])
			class_name = model.names[cls_id]

			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

			center_x = int((x1 + x2) / 2)
			center_y = int((y1 + y2) / 2)

			box_width = x2 - x1
			box_height = y2 - y1
			box_area = box_width * box_height

			if class_name in {"red", "red_buoy"}:
				red_buoys.append({
					"center": (center_x, center_y),
					"area": box_area,
					"box": (int(x1), int(y1), int(x2), int(y2))
				})

			elif class_name in {"green", "green_buoy"}:
				green_buoys.append({
					"center": (center_x, center_y),
					"area": box_area,
					"box": (int(x1), int(y1), int(x2), int(y2))
				})

		cv2.line(
			annotated_frame,
			(camera_center_x, 0),
			(camera_center_x, height),
			(255, 255, 0),
			2
		)

		cv2.line(
			annotated_frame,
			(0, camera_center_y),
			(width, camera_center_y),
			(255, 255, 0),
			1
		)

		cv2.circle(
			annotated_frame,
			(camera_center_x, camera_center_y),
			6,
			(255, 255, 0),
			-1
		)

		nearest_red = None
		nearest_green = None

		if len(red_buoys) > 0:
			nearest_red = max(red_buoys, key=lambda buoy: buoy["area"])

		if len(green_buoys) > 0:
			nearest_green = max(green_buoys, key=lambda buoy: buoy["area"])

		for buoy in red_buoys:
			x1, y1, x2, y2 = buoy["box"]
			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 180), 2)

		for buoy in green_buoys:
			x1, y1, x2, y2 = buoy["box"]
			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

		if nearest_red is not None:
			red_center = nearest_red["center"]
			x1, y1, x2, y2 = nearest_red["box"]

			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
			cv2.circle(annotated_frame, red_center, 8, (0, 0, 255), -1)

			cv2.putText(
				annotated_frame,
				"Nearest Red",
				(x1, y1 - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 0, 255),
				2
			)

		if nearest_green is not None:
			green_center = nearest_green["center"]
			x1, y1, x2, y2 = nearest_green["box"]

			cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
			cv2.circle(annotated_frame, green_center, 8, (0, 255, 0), -1)

			cv2.putText(
				annotated_frame,
				"Nearest Green",
				(x1, y1 - 10),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(0, 255, 0),
				2
			)

		if nearest_red is not None and nearest_green is not None:
			red_center = nearest_red["center"]
			green_center = nearest_green["center"]

			mid_x = int((red_center[0] + green_center[0]) / 2)
			mid_y = int((red_center[1] + green_center[1]) / 2)

			error_x = mid_x - camera_center_x

			cv2.line(
				annotated_frame,
				red_center,
				green_center,
				(255, 255, 255),
				3
			)

			cv2.circle(
				annotated_frame,
				(mid_x, mid_y),
				12,
				(255, 255, 255),
				-1
			)

			cv2.line(
				annotated_frame,
				(camera_center_x, camera_center_y),
				(mid_x, mid_y),
				(255, 0, 255),
				2
			)

			cv2.putText(
				annotated_frame,
				f"Nearest Gate Center: ({mid_x}, {mid_y})",
				(mid_x + 15, mid_y + 25),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.7,
				(255, 255, 255),
				2
			)

			cv2.putText(
				annotated_frame,
				f"Error X: {error_x}",
				(30, height - 70),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 0, 255),
				2
			)

			if abs(error_x) < 40:
				direction_text = "CENTER"
				direction_color = (0, 255, 0)

			elif error_x < 0:
				direction_text = "MOVE LEFT"
				direction_color = (0, 255, 255)

			else:
				direction_text = "MOVE RIGHT"
				direction_color = (0, 255, 255)

			cv2.putText(
				annotated_frame,
				direction_text,
				(30, height - 30),
				cv2.FONT_HERSHEY_SIMPLEX,
				1.0,
				direction_color,
				3
			)

			print(
				f"Nearest gate center: x={mid_x}, y={mid_y}, "
				f"error_x={error_x}, "
				f"red_area={nearest_red['area']:.2f}, "
				f"green_area={nearest_green['area']:.2f}"
			)

		else:
			missing = []
			if nearest_red is None:
				missing.append("red")
			if nearest_green is None:
				missing.append("green")
			missing_text = ", ".join(missing)

			cv2.putText(
				annotated_frame,
				f"Gate not found: missing {missing_text}",
				(30, height - 70),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(0, 255, 255),
				2
			)

		cv2.imshow("YOLO Nearest Red-Green Buoy Gate", annotated_frame)

		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
