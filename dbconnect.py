import MySQLdb

def connection():
    conn = MySQLdb.connect(host="localhost",
                            user = "root",
                            passwd = "Password123*",
                            db = "flask_webapp")
    
    c = conn.cursor()

    return c, conn