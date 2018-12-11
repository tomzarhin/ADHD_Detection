from flask import render_template
from flask import Flask

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=('GET', 'POST'))
def main():
    return render_template('home.html')

@app.route("/Try")
def tryingToPrintTom():
    print("try")
    return "Tom!"

@app.route("/param/<name>")
def asd(name):
    print("try")
    return "Tom!"

if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)
