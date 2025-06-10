import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import xlsxwriter as xw
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from rembg import remove
import pickle
from ultralytics import YOLO
import threading

# Global variables
global fileImage, resize_img, model, is_running, cap, selected_image_path
fileImage = None
resize_img = None
is_running = False
cap = None
selected_image_path = None

# Load the trained model
with open('model 1.pkl', 'rb') as r:
    model = pickle.load(r)

# Disable chained assignment warning in pandas
pd.set_option('mode.chained_assignment', None)

# Initialize Tkinter window
window = tk.Tk()
window.configure(bg='#B0C4DE')
window.geometry("700x600")
window.resizable(False, False)
window.title("KLASIFIKASI KUALITAS TELUR BURUNG PUYUH")

# Quit app when Escape key is pressed
window.bind('<Escape>', lambda e: window.quit())

# 🔍 Logika Pengukuran Jarak

KNOWN_DISTANCE = 100  # cm
KNOWN_WIDTH = 3       # cm (lebar telur nyata)
BOX_PIXEL_WIDTH_AT_KNOWN_DISTANCE = 50  # hasil kalibrasi manual saat jarak 1m

focal_length = (BOX_PIXEL_WIDTH_AT_KNOWN_DISTANCE * KNOWN_DISTANCE) / KNOWN_WIDTH

def estimate_distance(pixel_width):
    if pixel_width == 0:
        return float('inf')
    return (KNOWN_WIDTH * focal_length) / pixel_width

# 💡 Normalisasi Cahaya
def normalize_brightness(image, target_mean=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    scale_factor = target_mean / (mean_intensity + 1e-6)
    normalized = cv2.convertScaleAbs(image, alpha=scale_factor, beta=0)
    return normalized

# 🌟 Deteksi Kondisi Pencahayaan
def check_light_condition(image, threshold_low=40, threshold_high=200):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity < threshold_low:
        return "Redup"
    elif mean_intensity > threshold_high:
        return "Terang"
    else:
        return "Ideal"

# Function to show real-time webcam feed
def webcam_capture():
    def run():
        # Coba akses kamera di index 2 dulu
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("⚠️ Index 2 tidak bisa dibuka. Mencoba index 1...")
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("⚠️ Index 1 juga tidak bisa dibuka. Menggunakan index 0...")
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("❌ Tidak ada kamera tersedia.")
                    return
        detect = YOLO('modelyolo/best.pt')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detect(frame)[0]
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                width_pixel = x2 - x1
                distance = estimate_distance(width_pixel)

                # Hanya lanjutkan jika telur dalam jarak <= 1 meter
                if distance <= 100:
                    kotak = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    img = frame[int(y1):int(y2), int(x1):int(x2)]
                    resized_frame = cv2.resize(img, (224, 224))

                    # Normalisasi pencahayaan
                    normalized_frame = normalize_brightness(resized_frame)
                    light_status = check_light_condition(normalized_frame)

                    data = extract_features(normalized_frame)
                    data_df = pd.DataFrame([data])
                    predictions = model.predict(data_df)[0]

                    label = f"{predictions} | {distance:.0f} cm | {light_status}"
                    color = (0, 255, 0) if "Ideal" in light_status else (0, 165, 255)
                    cv2.putText(kotak, label, (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                else:
                    # Tampilkan kotak merah jika terlalu jauh
                    kotak = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame, "Jauh", (int(x1), int(y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Klasifikasi Kualitas Telur', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.getWindowProperty('Klasifikasi Kualitas Telur', cv2.WND_PROP_VISIBLE) < 1:
                break

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=run, daemon=True).start()

# Function to open and display an image
def openImage():
    global selected_image_path
    file_path = filedialog.askopenfilename(
        title="Pilih Gambar",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    try:
        preprocessed_path = preprocess_image(file_path)
        img = Image.open(preprocessed_path).convert("RGBA")
        img_tk = ImageTk.PhotoImage(img)
        frame_asli.config(image=img_tk)
        frame_asli.image = img_tk
        selected_image_path = preprocessed_path
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Gagal membaca gambar.")
    output_image = remove(image)
    preprocessed_path = "removed_background.png"
    cv2.imwrite(preprocessed_path, output_image)
    pil_image = Image.open(preprocessed_path).convert("RGBA")
    mask = pil_image.split()[-1]
    mask_np = np.array(mask)
    _, binary_mask = cv2.threshold(mask_np, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Tidak ada objek yang ditemukan setelah menghapus background.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = pil_image.crop((x, y, x + w, y + h))
    fixed_size = (300, 300)
    cropped_image_resized = cropped_image.resize(fixed_size, Image.Resampling.LANCZOS)
    preprocessed_cropped_path = "preprocessed_image.png"
    cropped_image_resized.save(preprocessed_cropped_path)
    return preprocessed_cropped_path

def ekstrak_ciri():
    nilai_table = []
    values = [value for value in feature_df.iloc[0]]
    for value in values:
        nilai_table.append(value)
    prediction = model.predict(feature_df)[0]
    y = 50
    x = 67
    for i in range(18):
        if i != 0 and i % 3 == 0:
            x = 67
            y += 25
        label_j = tk.Label(frame_ekstrak, background='white', highlightbackground="black",
                           text=round(nilai_table[i], 2), width=10, height=1, highlightthickness="1")
        label_j.place(x=x, y=y)
        x += 80
    output_label.config(text=prediction)

def TrainingData():
    global feature_df
    if 'selected_image_path' not in globals():
        messagebox.showerror("Error", "Silakan pilih gambar terlebih dahulu.")
        return
    try:
        features = extract_features(selected_image_path)
        feature_df = pd.DataFrame([features])
        messagebox.showinfo("Info", "Fitur berhasil diekstraksi.")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

# Fungsi baru: Konversi RGB ke HSI
def rgb_to_hsi(rgb_image):
    rgb = rgb_image.astype('float32') / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    intensity = (r + g + b) / 3.0
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = 1 - 3 * min_rgb / (r + g + b + 1e-8)
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g)**2 + (r - b)*(g - b)) + 1e-8
    theta = np.arccos(numerator / denominator)
    hue = np.zeros_like(theta)
    hue[g >= b] = theta[g >= b]
    hue[g < b] = 360 - theta[g < b]
    hue = hue * 360 / (2 * np.pi)
    hue = np.clip(hue, 0, 360).astype(np.uint8)
    saturation = np.clip(saturation, 0, 1).astype(np.float32) * 255
    intensity = intensity * 255
    return hue, saturation, intensity

# Function to extract features from image
def extract_features(image_input):
    if isinstance(image_input, str):
        gambar = cv2.imread(image_input, cv2.IMREAD_UNCHANGED)
        if gambar is None:
            raise ValueError("Gagal membaca gambar dari alamat file.")
    elif isinstance(image_input, np.ndarray):
        gambar = image_input
    else:
        raise TypeError("Input harus berupa alamat file (string) atau array NumPy.")

    new_size = (224, 224)
    image = cv2.resize(gambar, new_size)

    blue_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    red_channel = image[:, :, 2]

    mean_red, variance_red, range_red = calculate_features(red_channel)
    mean_green, variance_green, range_green = calculate_features(green_channel)
    mean_blue, variance_blue, range_blue = calculate_features(blue_channel)

    hue_channel, saturation_channel, intensity_channel = rgb_to_hsi(image)

    mean_hue, variance_hue, range_hue = calculate_features(hue_channel)
    mean_saturation, variance_saturation, range_saturation = calculate_features(saturation_channel)
    mean_intensity, variance_intensity, range_intensity = calculate_features(intensity_channel)

    features = {
        "Mean Red": mean_red,
        "Variance Red": variance_red,
        "Range Red": range_red,
        "Mean Green": mean_green,
        "Variance Green": variance_green,
        "Range Green": range_green,
        "Mean Blue": mean_blue,
        "Variance Blue": variance_blue,
        "Range Blue": range_blue,
        "Mean Hue": mean_hue,
        "Variance Hue": variance_hue,
        "Range Hue": range_hue,
        "Mean Saturation": mean_saturation,
        "Variance Saturation": variance_saturation,
        "Range Saturation": range_saturation,
        "Mean Intensity": mean_intensity,
        "Variance Intensity": variance_intensity,
        "Range Intensity": range_intensity
    }
    return features

def calculate_features(channel):
    mean_val = np.mean(channel)
    variance_val = np.var(channel)
    range_val = np.max(channel) - np.min(channel)
    return mean_val, variance_val, range_val

# GUI Layout
frame_judul = tk.Frame(window, width=660, height=60, background='white', highlightbackground="black", highlightthickness="1")
frame_judul.place(x=20, y=20)
label_judul = tk.Label(frame_judul, text="KLASIFIKASI KUALITAS TELUR BURUNG PUYUH MENGGUNAKAN METODE NAÏVE BAYES CLASSIFIER", font=("Cambria Bold", 10), fg="black", background="white")
label_judul.place(x=25, y=20)

placeholder = Image.new("RGB", (300, 300), color="#cccccc")
draw = ImageDraw.Draw(placeholder)
draw.text((100, 140), "Gambar Akan Ditampilkan Disini", fill="black")
placeholder_tk = ImageTk.PhotoImage(placeholder)
frame_asli = tk.Label(window, background='white', width=301, height=301, highlightbackground="black", highlightthickness="1", image=placeholder_tk)
frame_asli.place(x=20, y=120)
label_asli = tk.Label(window, text="Citra Asli", font=("Cambria Bold", 12), fg="black", background="#B0C4DE")
label_asli.place(x=25, y=95)

frame_ekstrak = tk.Frame(window, background='#B0C4DE', width=330, height=300, highlightbackground="black", highlightthickness="1")
frame_ekstrak.place(x=350, y=120)
label_ekstrak = tk.Label(window, text="Ekstraksi gambar", font=("Cambria Bold", 12), fg="black", background="#B0C4DE")
label_ekstrak.place(x=355, y=95)

# Buttons
tk.Button(window, text="Buka gambar", command=openImage, height=1, width=15).place(x=115, y=430)
tk.Button(window, text="Training data", command=TrainingData, height=1, width=15).place(x=115, y=470)
tk.Button(window, text="Ekstrak", command=ekstrak_ciri, height=1, width=15).place(x=465, y=430)
tk.Button(window, text="Realtime", command=webcam_capture, height=1, width=15).place(x=465, y=510)

# Output label
output_label = tk.Label(window, font=('Cambria Bold', 11), highlightbackground="black", highlightthickness="1", background='white', width=22)
output_label.place(x=420, y=470)

# Table layout
kolom = ['', 'Red', 'Green ', 'Blue', 'H', 'S', 'I']
baris = ['', 'Mean', 'Variance', 'Range']
y = 0
x = 0
for i in range(7):
    y += 25
    label_i = tk.Label(frame_ekstrak, background='grey', text=kolom[i], highlightbackground="black", width=5, height=1, highlightthickness="1")
    label_i.place(x=22, y=y)
    if i > 0 and i < 4:
        if i == 1:
            x = 67
        label_j = tk.Label(frame_ekstrak, background='grey', text=baris[i], highlightbackground="black", width=10, height=1, highlightthickness="1")
        label_j.place(x=x, y=25)
        x += 80
y = 50
x = 67
for i in range(18):
    if i == 6 or i == 12:
        x += 80
        y = 50
    label_j = tk.Label(frame_ekstrak, background='white', highlightbackground="black", width=10, height=1, highlightthickness="1")
    label_j.place(x=x, y=y)
    y += 25

# Run the application
window.mainloop()
