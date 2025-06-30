import pickle
from flask import  Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app=Flask(__name__)
#Load the Model
model = pickle.load(open("RFCmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("home.html")

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print("data========",data)
#     print("erray=========",np.array(list(data.values())).reshape(1, -1))
#     new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
#     output = regmodel.predict(new_data)
#     print("output============",output[0])
#     return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print("data****************", data)

    # map: form field -> encoder key
    form_to_encoder = {
        'CustomerName': 'Customer Name',
        'CustomerEmail': 'Customer Email',
        'CustomerGender': 'Customer Gender',
        'ProductPurchased': 'Product Purchased',
        'TicketType': 'Ticket Type',
        'TicketSubject': 'Ticket Subject',
        'TicketStatus': 'Ticket Status',
        'Resolution': 'Resolution',
        'TicketPriority': 'Ticket Priority',
        'TicketChannel': 'Ticket Channel'
    }

    numerical_features = ['CustomerAge', 'Purchase_Year', 'Purchase_Month', 'Response_Hour', 'Resolution_Hour']

    encoded_cats = []
    for form_field, encoder_key in form_to_encoder.items():
        val = data.get(form_field)
        if val is None:
            print(f"Missing value for field: {form_field}")
            val = ""
        val = val.strip()  # clean
        encoder = encoders[encoder_key]
        try:
            encoded_val = encoder.transform([val])[0]
        except ValueError:
            print(f"Unseen label: '{val}' for field: '{form_field}'")
            encoded_val = -1  # fallback
        encoded_cats.append(encoded_val)

    numerical_vals = [float(data[col]) for col in numerical_features]

    final_input = np.array(encoded_cats + numerical_vals).reshape(1, -1)
    scaled_input = scaler.transform(final_input)
    print("final input =====", scaled_input)
    output = model.predict(scaled_input)[0]

    return render_template("home.html", prediction_text="The Customer Satisfaction Prediction Rating is- {}".format(output))


if __name__=="__main__":
    app.run(debug=True)