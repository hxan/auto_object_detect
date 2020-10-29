import tkinter as tk  
import tkinter.font as tf
from tkinter.filedialog import (askdirectory,askopenfilenames,askopenfilename,asksaveasfilename)
import commands
import subprocess
windows = tk.Tk()
windows.title('My Window')
windows.geometry('600x280')  


def face_start():
    subprocess.Popen('./face')
def face_stop():
    subprocess.Popen('killall face',shell=True)
Button1=tk.Button(windows,text='Face Start',font=tf.Font(size=40),command=face_start)
Button1.place(height = 50,width = 100,x = 10,y = 14)
Button2=tk.Button(windows,text='Face Stop',font=tf.Font(size=40),command=face_stop)
Button2.place(height = 50,width = 100,x = 10,y = 64)


def rect_slient_start():
    path_=askopenfilename()
    subprocess.Popen('./slient_rect_ '+path_,shell=True)
def rect_slient_stop():
    subprocess.Popen('killall slient_rect_',shell=True)
Button3=tk.Button(windows,text='Rect Slient Start',font=tf.Font(size=40),command=rect_slient_start)
Button3.place(height = 50,width = 120,x = 10,y = 134)
Button4=tk.Button(windows,text='Rect Slient Stop',font=tf.Font(size=40),command=rect_slient_stop)
Button4.place(height = 50,width = 120,x = 10,y = 184)

def rect_move_start():
    subprocess.Popen('./rect_move')
def rect_move_stop():
    subprocess.Popen('killall rect_move',shell=True)
Button5=tk.Button(windows,text='Rect Move Start',font=tf.Font(size=40),command=rect_move_start)
Button5.place(height = 50,width = 120,x = 180,y = 14)
Button6=tk.Button(windows,text='Rect Move Stop',font=tf.Font(size=40),command=rect_move_stop)
Button6.place(height = 50,width = 120,x = 180,y = 64)



def obj_move_start():
    subprocess.Popen('python ./object_detect_move_py/object_detect_move.py ',shell=True)
def obj_move_stop():
    subprocess.Popen('killall python',shell=True)
Button9=tk.Button(windows,text='Object Move Start',font=tf.Font(size=40),command=obj_move_start)
Button9.place(height = 50,width = 120,x = 180,y = 134)
Button10=tk.Button(windows,text='Object MOve Stop',font=tf.Font(size=40),command=obj_move_stop)
Button10.place(height = 50,width = 120,x = 180,y = 184)


def network_start():
    subprocess.Popen('source activate tensorflow;python ./reco.py',shell=True)
def network_stop():
    subprocess.Popen('killall python',shell=True)
Button12=tk.Button(windows,text='Network Start',font=tf.Font(size=40),command=network_start)
Button12.place(height = 50,width = 120,x =320,y = 134)
Button13=tk.Button(windows,text='Network Stop',font=tf.Font(size=40),command=network_stop)
Button13.place(height = 50,width = 120,x =320,y = 184)

windows.mainloop()

