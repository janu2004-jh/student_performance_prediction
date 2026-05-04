from flask import Flask, render_template, request, redirect, url_for, session
import pickle

app = Flask(__name__)
app.secret_key = "secret123"   # 🔥 required for session

model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    # get result once and clear it
    result = session.pop("prediction_text", None)
    return render_template("index.html", prediction_text=result)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        previous_marks = float(request.form["previous_marks"])

        data = [[study_hours, attendance, previous_marks]]

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0]

        confidence = round(max(prob) * 100, 2)

        result = "✅ Pass" if prediction == 1 else "❌ Fail"

        # 🔥 store temporarily instead of rendering directly
        session["prediction_text"] = f"{result} ({confidence}%)"

        # 🔥 redirect to home (fixes refresh issue)
        return redirect(url_for("home") + "#predict")

    except Exception as e:
        print(e)
        session["prediction_text"] = "Error"
        return redirect(url_for("home") + "#predict")

if __name__ == "__main__":
    app.run(debug=True)