import tkinter as tk
from tkinter import *
from tkinter import filedialog
import pandas as pd
import xlsxwriter as xw
from PIL import Image,ImageTk
import cv2
import numpy as np
import converter
from rembg import remove
import pickle
from ultralytics import YOLO


global model 

with open ('model78','rb') as r:
    model = pickle.load(r)

pd.set_option('mode.chained_assignment', None)

window = tk.Tk()


window.configure (bg='#B0C4DE')
window.geometry("700x600")
window.resizable(False,False)
window.title ("KLASIFIKASI KUALITAS TELUR BURUNG PUYUH ")

# # quit app whenever pressed 
window.bind('<Escape>', lambda e: window.quit())

def show_frame():
    cap = cv2.VideoCapture(0)
    class_labels = ['Bagus','Buruk','Sedang']
    detect = YOLO('model/best.pt')
    cap.set(cv2.CAP_PROP_SETTINGS,1)

    while True :
     #capture Frame by frame
        ret, frame = cap.read()
        ret, image = cap.read()
        results = detect(image)[0]
    # Make a prediction
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            kotak = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            img = image[int(y1):int(y2),int(x1):int(x2)]
            resized_frame = cv2.resize(img, (224, 224))
            hsi = converter.RGB_TO_HSI(resized_frame)
            b,g,r = cv2.split(resized_frame)
            i,s,h = cv2.split(hsi)

            m_red=np.nanmean(np.array(r))
            m_green=np.nanmean(np.array(g))
            m_blue =np.nanmean(np.array(b))

            v_red = np.nanstd(np.array(r))
            v_green = np.nanstd(np.array(g))
            v_blue = np.nanstd(np.array(b))

            r_red = np.nanmax(np.array(r)) - np.nanmin(np.array(r))
            r_green = np.nanmax(np.array(g)) - np.nanmin(np.array(g))
            r_blue = np.nanmax(np.array(b)) - np.nanmin(np.array(b))
            
            m_h = np.nanmean(np.array(h))
            m_s = np.nanmean(np.array(s))
            m_i = np.nanmean(np.array(i))
                
            v_h = np.nanstd(np.array(h))
            v_s = np.nanstd(np.array(s))
            v_i = np.nanstd(np.array(i))

            r_h= np.nanmax(np.array(h)) - np.nanmin(np.array(h))
            r_s= np.nanmax(np.array(s)) - np.nanmin(np.array(s))
            r_i = np.nanmax(np.array(i)) - np.nanmin(np.array(i))

            h = ['mean_red','mean_green','mean_blue','variance_red','variance_green','variance_blue','range_red','range_green','range_blue','mean_h','mean_s','mean_i', 'variance_h','variance_s', 'variance_i','range_h','range_s','range_i']
            f = [m_red,m_green,m_blue,v_red,v_green,v_blue,r_red,r_green,r_blue,m_h,m_s,m_i,v_h,v_s,v_i,r_h,r_s,r_i]
            gabung= pd.DataFrame({h[0]:[f[0]],h[1]:[f[1]],h[2]:[f[2]],h[3]:[f[3]],h[4]:[f[4]],h[5]:[f[5]],h[6]:[f[6]],h[7]:[f[7]],h[8]:[f[8]],h[9]:[f[9]],h[10]:[f[10]],h[11]:[f[11]],h[12]:[f[12]],h[13]:[f[13]],h[14]:[f[14]],h[15]:[f[15]],h[16]:[f[16]],h[17]:[f[17]]}) 
            predictions = model.predict(gabung)
            label = f"{class_labels[predictions[0]]}"
            cv2.putText(kotak,label, (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Klasifikasi Kualitas Telur', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('Klasifikasi Kualitas Telur',cv2.WND_PROP_VISIBLE) < 1:        
            break
    
    cap.release()
    cv2.destroyAllWindows()



def openImage():
    global fileImage
    global resize_img

    for widget in frame_asli.winfo_children():
        widget.destroy()

    fileImage = filedialog.askopenfilename()
    img = Image.open(fileImage)
    reimg =remove(img)
    resize_img = reimg.resize((300, 300))
    photo = ImageTk.PhotoImage(image = resize_img )
    label = tk.Label(frame_asli, image = photo)
    label.image = photo
    label.place(x=-0.5,y=-0.5)
    
    

def ekstrak_ciri():
    y=50
    x=67
    for i in range (18):
        if i==6 or i==12:
            x+=80
            y=50
        label_j = tk.Label(frame_ekstrak,background='white',highlightbackground="black",text=round(nilai_table[i],2),width = 10, height = 1, highlightthickness="1")
        label_j.place(x=x,y=y)
        y+=25
    hasil_ekstrak = pd.read_excel('databaru.xlsx',sheet_name='Sheet1')
    print(hasil_ekstrak)
    hasil_klasifikasi = model.predict(hasil_ekstrak)
    if hasil_klasifikasi[0]==0:
        Ouput = tk.Label(window, font=('Cambria Bold',11),highlightbackground="black", highlightthickness="1",text="Telur kualitas bagus", background='white', width= 22).place(x=420,y=470)
    if hasil_klasifikasi[0]==1:
         Ouput = tk.Label(window, font=('Cambria Bold',11),highlightbackground="black", highlightthickness="1",text="Telur kualitas buruk", background='white', width= 22).place(x=420,y=470)
    if hasil_klasifikasi[0]==2:
        Ouput = tk.Label(window, font=('Cambria Bold',11),highlightbackground="black", highlightthickness="1",text="Telur kualitas sedang", background='white', width= 22).place(x=420,y=470)

def TrainingData():
    global nilai_table
    img = Image.open(fileImage)
    rmv = remove(img)
    reimg = rmv.resize((224,224))
    creimg = reimg.convert('RGB')
    gambar = np.array(creimg)
    hsi = converter.RGB_TO_HSI(gambar)
    b,g,r = cv2.split(gambar)
    i,s,h = cv2.split(hsi)

    m_red=np.nanmean(np.array(r))
    m_green=np.nanmean(np.array(g))
    m_blue =np.nanmean(np.array(b))

    v_red = np.nanstd(np.array(r))
    v_green = np.nanstd(np.array(g))
    v_blue = np.nanstd(np.array(b))

    r_red = np.nanmax(np.array(r)) - np.nanmin(np.array(r))
    r_green = np.nanmax(np.array(g)) - np.nanmin(np.array(g))
    r_blue = np.nanmax(np.array(b)) - np.nanmin(np.array(b))
   
    m_h = np.nanmean(np.array(h))
    m_s = np.nanmean(np.array(s))
    m_i = np.nanmean(np.array(i))
    
    v_h = np.nanstd(np.array(h))
    v_s = np.nanstd(np.array(s))
    v_i = np.nanstd(np.array(i))

    r_h= np.nanmax(np.array(h)) - np.nanmin(np.array(h))
    r_s= np.nanmax(np.array(s)) - np.nanmin(np.array(s))
    r_i = np.nanmax(np.array(i)) - np.nanmin(np.array(i))

    nilai_table = [m_red,m_green,m_blue,m_h,m_s,m_i,v_red,v_green,v_blue,v_h,v_s,v_i,r_red,r_green,r_blue,r_h,r_s,r_i]
    book = xw.Workbook('databaru.xlsx')
    sheet = book.add_worksheet()

    kolom = 0
    # Kolom Feature RGB
    rgb_feature = ['mean_red','mean_green','mean_blue', 'variance_red', 'variance_green', 'variance_blue','range_red','range_green','range_blue']
    for i in rgb_feature :
        sheet.write (0,kolom,i)
        kolom+=1

    # Kolom Feature HSI
    hsi_feature = ['mean_h','mean_s','mean_i', 'variance_h', 'variance_s', 'variance_i','range_h','range_s','range_i']
    for i in hsi_feature:
        sheet.write(0,kolom,i)
        kolom+=1

    feature_rgb= [m_red,m_green,m_blue,v_red,v_green,v_blue,r_red,r_green,r_blue]
    feature_hsi= [m_h,m_s,m_i,v_h,v_s,v_i,r_h,r_s,r_i]
    
    column = 0
    for item in feature_rgb:
        sheet.write (1,column,round(item,3))
        column+=1
    for item in feature_hsi:    
        sheet.write (1,column,round(item,3))
        column+=1

    book.close()



#Frame Judul
frame_judul = tk.Frame (window,width=660, height=60, background='white',highlightbackground="black", highlightthickness="1")
frame_judul.place(x=20, y=20)
label_judul = tk.Label (frame_judul, text="KLASIFIKASI KUALITAS TELUR BURUNG PUYUH MENGGUNAKAN METODE NAÏVE BAYES CLASSIFIER", font=("Cambria Bold",10),fg="black", background="white")
label_judul.place(x=25, y=20)
label = tk.Label (window, text="Darul", font=("Cambria Bold",9),fg="black", background="#B0C4DE")
label.place( x=330, y=580)

#Frame gambar Asli
frame_asli = tk.Frame(window, background='white',width=301, height=301,highlightbackground="black", highlightthickness="1")
frame_asli.place(x=20, y=120)
label_asli = tk.Label (window, text="Citra Asli", font=("Cambria Bold",12),fg="black", background="#B0C4DE")
label_asli.place( x=25, y=95)

#frame Ekstrak Ciri
frame_ekstrak = tk.Frame(window, background='#B0C4DE',width=330, height=300,highlightbackground="black", highlightthickness="1")
frame_ekstrak.place(x=350, y=120)
label_ekstrak = tk.Label (window, text="Ekstraksi gambar", font=("Cambria Bold",12),fg="black", background="#B0C4DE")
label_ekstrak.place( x=355, y=95)

kolom = ['','Red','Green ','Blue', 'H', 'S','I']
baris = ['','Mean','Variance','Range']
y=0
x=0

for i in range (7):
    y +=25
    label_i = tk.Label(frame_ekstrak,background='grey', text=kolom[i] ,highlightbackground="black",width = 5, height = 1, highlightthickness="1")
    label_i.place(x=22,y=y)
    if i >0 and i<4:
        if i==1:
            x=67
        label_j = tk.Label(frame_ekstrak,background='grey', text=baris[i] ,highlightbackground="black",width = 10, height = 1, highlightthickness="1")
        label_j.place(x=x,y=25)
        x+=80

y=50
x=67
for i in range (18):
    if i==6 or i==12:
        x+=80
        y=50
    label_j = tk.Label(frame_ekstrak,background='white',highlightbackground="black",width = 10, height = 1, highlightthickness="1")
    label_j.place(x=x,y=y)
    y+=25

#Tombol Buka Gambar
open_image = tk.Button (window, text="Buka gambar", command = openImage, height=1,width=15).place(x=115,y=430)

#Tombol Buka Gambar
training_data = tk.Button (window, text="Training data", command = TrainingData, height=1,width=15).place(x=115,y=470)

#Tombol Ekstrak feature
ekstrak_citra= tk.Button (window, text="Ekstrak", command= ekstrak_ciri, height=1,width=15).place (x=465, y=430)
Ouput = tk.Label(window, font=('Cambria Bold',11),highlightbackground="black", highlightthickness="1",text="", background='white', width= 22).place(x=420,y=470)

#Tombol Realtime
Realtime= tk.Button (window, text="Realtime", command= show_frame, height=1,width=15).place (x=465, y=510)

window.mainloop()