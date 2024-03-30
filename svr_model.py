from flask import Flask, request, render_template
from text2img_model import create_pipeline, text2img

# Khởi tạo Flask app
app =  Flask(__name__)

# Định nghĩa tham số
IMAGE_PATH  = "static/output.jpg"

# Khởi tạo pipeline
pipeline = create_pipeline()

@app.route("/", methods = ['GET', 'POST'])
def index():
    # Kiểm tra xem có phải người dùng view trang web không
    if request.method == "GET":
        # Trả về giao diện trang web
        return render_template("index.html")
    else:
        # Xử lý việc người dùng submit prompt-> sinh ảnh -> trả về
        user_input = request.form["prompt"]

        print("Start gen....")
        im = text2img(user_input, pipeline)
        print("Finish gen....")
        im.save(IMAGE_PATH)

        return render_template("index.html", image_url = IMAGE_PATH)

if __name__ =='__main__':
    app.run(debug=False, host='0.0.0.0', port=8888, use_reloader=False)


