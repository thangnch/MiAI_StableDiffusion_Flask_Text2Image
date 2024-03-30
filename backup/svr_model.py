from flask import Flask, request, render_template, flash, send_file
from text2img_model import text2img, create_pipeline

app = Flask(__name__)
IMAGE_PATH = "../static/output.jpg"

pipeline  = create_pipeline()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    else:
        user_input = request.form.get("prompt")

        print("Start to gen...")
        im = text2img(prompt=user_input, pipeline=pipeline)
        im.save(IMAGE_PATH)
        print("Finish gen...")

        return render_template(
            "index.html",
            image_url=IMAGE_PATH,
        )


# Thuc thi server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=8888, use_reloader=False)