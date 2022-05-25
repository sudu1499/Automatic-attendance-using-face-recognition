from cProfile import label
from tkinter import *
from utils.live_face_recognition import live_recognize,stop

window=Tk()
window.title('Attendance Monitoring system')
window.configure(bg='black')
window.geometry('800x600')
b2=Button(window,text='start detecting',width=20,height=2,borderwidth=2,command=live_recognize).place(x=50,y=250)
b3=Button(window,text='stop',width=20,height=2,borderwidth=2,command=stop).place(x=50,y=350)
window.mainloop()
