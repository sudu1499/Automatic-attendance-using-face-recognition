from tkinter import Button
from flask import *
import cv2
import sqlite3
import base64
import os
from glob import glob
import yaml
import dlib

app = Flask(__name__)  

@app.route('/login',methods = ['GET','POST'])  
def login(): 
    if request.method=="POST": 
      uname=request.form['uname']  
      passwrd=request.form['pass']  
      if uname=="Tejas" and passwrd=="1234":  
          return render_template("mainpage.html")
      else:
          return "<h1 style=color:red>Invalid user</h1>"
    else:
        return render_template("login.html") 
 

@app.route('/logout') 
def logout():
    return render_template("login.html")  

@app.route('/ack',methods = ['GET','POST'])
def ack():
    global tname
    if request.method=="POST": 
        tname=request.form['sname']  
        print(tname)
        config=yaml.safe_load(open(r'D:\projects\new_face_recognition\utils\config.yaml','r'))
        img_path=config['img_path']
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("Registration")
        os.makedirs(img_path+'\\'+tname,exist_ok=True)
        f_path=img_path+'\\'+tname+'\\'
        c=0
        det=dlib.get_frontal_face_detector()
        while c<100:
            ret, frame = cam.read()
            gframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            d=det(gframe)
            if len(d):
                for i in d :
                    crp=frame[i.top():i.bottom(),i.left():i.right()]
                    cv2.imwrite(f_path+f'{c}'+'.jpeg',crp)

                c+=1
                cv2.imshow("Registration", frame)
            else:
                cv2.destroyAllWindows() 
            if cv2.waitKey(1)==ord('q'):
                cv2.destroyAllWindows()
                break
        cam.release()
        cv2.destroyAllWindows()

        # for i in glob(f_path+'*'):
        #     with open(i,'rb') as f:
        #         data=f.read()
        #         database(i.split('\\')[-2],data)
        for i in glob(f_path+'*'):
            img_str=cv2.imencode('.jpeg',cv2.imread(i))[1].tostring()
            database(i.split('\\')[-2],img_str)

        config['no_students']+=1
        yaml.dump(config,open(r'D:\projects\new_face_recognition\utils\config.yaml','w'))
        return render_template('mainpage.html')
        
def database(name,m):
    config=yaml.safe_load(open(r'D:\projects\new_face_recognition\utils\config.yaml','r'))
    student_face_db=config['student_face_db']
    conn= sqlite3.connect(student_face_db)
    cursor = conn.cursor()

    cursor.execute("""CREATE TABLE IF NOT EXISTS student_face ( name TEXT, m BLOP) """)
    cursor.execute("""INSERT INTO student_face (name,m) VALUES(?,?) """,(name,m))
    print(cursor.fetchall())
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':  
   app.run(debug = True)  
 