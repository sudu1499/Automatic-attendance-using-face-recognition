import sqlite3

conn=sqlite3.connect('attendance.db')
c=conn.cursor()
c.execute("create table student(name text,date text)")
conn.commit()

