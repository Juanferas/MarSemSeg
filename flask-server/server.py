from flask import Flask, request
import os

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/files"

# Test API route
@app.route("/test")
def test():
    return {"test": "It works!"}

@app.route("/file-upload", methods=["POST"])
def file_upload():
    if request.files:

        file = request.files["myFile"]
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config["UPLOAD_FOLDER"], file.filename))

        print("File saved :)")

        return {"success": True}

if __name__ == "__main__":
    app.run(debug=True)
