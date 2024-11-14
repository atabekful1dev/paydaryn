import cv2
import os
from flask import Flask, render_template, Response, redirect, url_for

app = Flask(__name__)

# Путь к эталонному изображению
REFERENCE_PATH = 'reference.jpg'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Загружаем эталонное изображение и находим лицо
reference_image = cv2.imread(REFERENCE_PATH, cv2.IMREAD_GRAYSCALE)
reference_face = None
faces = face_cascade.detectMultiScale(reference_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in faces:
    reference_face = reference_image[y:y + h, x:x + w]

# Проверка наличия лица на эталонном изображении
if reference_face is None:
    raise Exception("Лицо на эталонном изображении не найдено. Загрузите изображение с лицом.")

# Порог для подтверждения лица
THRESHOLD = 50

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Обработчик видеопотока для камеры
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        recognized = False
        for (x, y, w, h) in detected_faces:
            current_face = gray[y:y + h, x:x + w]
            diff = cv2.absdiff(reference_face, cv2.resize(current_face, reference_face.shape[::-1]))
            score = int(cv2.mean(diff)[0])

            # Проверка, распознано ли лицо
            if score < THRESHOLD:
                recognized = True

            # Рисуем прямоугольник вокруг лица
            color = (0, 255, 0) if recognized else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if recognized:
            cap.release()
            cv2.destroyAllWindows()
            return redirect(url_for('result', status='success'))
    cap.release()

# Маршрут для видеопотока
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Маршрут для результата
@app.route('/result/<status>')
def result(status):
    if status == 'success':
        message = "Лицо распознано. Оплата подтверждена!"
    else:
        message = "Лицо не распознано. Оплата отклонена."
    return render_template('result.html', message=message, status=status)


if __name__ == "__main__":
    app.run(debug=True)
