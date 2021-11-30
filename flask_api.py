from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# Load the models/scaler from disk
reg_model = pickle.load(open("reg_model.pickle", 'rb'))
lstm_scaler = pickle.load(open("lstm_scaler.pickle", 'rb'))
lstm_model = load_model('lstm_model.h5')


@app.route("/")
@cross_origin()
def helloWorld():
    return "Backend running!"


@app.route("/prediction_reg", methods=["GET"])
def predTests():

    if request.method == "GET":
        # Receive data from the frontend
        daily_cases = float(request.args.get('daily_cases'))
        daily_recoveries = float(request.args.get('daily_recoveries'))
        daily_deaths = float(request.args.get('daily_deaths'))

        # Format data and predict
        arguments = [[daily_cases, daily_recoveries, daily_deaths]]
        result = reg_model.predict(arguments).astype(int)[0]

        return jsonify(str(result))

@app.route("/prediction_lstm", methods=["GET"])
def predLSTM():

    if request.method == "GET":
        # Receive data from the frontend
        # daily_cases = float(request.args.get('daily_cases'))
        
        
        # Due to the huge amount of input data (1, 14, 4) necessary to 
        # predict the next value the API just returns a random example:

        # Columns:   Daily cases, Daily reco,  Daily deaths, Daily tests
        arguments = [[[-0.74709421, -0.61016812, -1.26282585,  0.15429546],
                    [-0.69556239, -0.58789429, -1.23030558,  0.23374378],
                    [-0.69459789, -0.56896702, -0.92678311,  0.23001341],
                    [-0.6954935 , -0.66129917, -1.10022452,  0.10104652],
                    [-0.6188158 , -0.63886076, -1.14358488, -0.38435339],
                    [-0.66214938, -0.61773902, -1.00266373, -0.62856284],
                    [-0.5632882 , -0.60967436, -1.07854435, -0.11266061],
                    [-0.45939785, -0.60298124, -0.99182364, -0.0201473 ],
                    [-0.3707328 , -0.59376448, -1.01350382,  0.01342608],
                    [-0.24335   , -0.60320069, -1.04602408,  0.05461636],
                    [-0.06567546, -0.58268243, -0.95930338,  0.06087993],
                    [ 0.13769611, -0.49912814, -1.08938444,  0.45040052],
                    [ 0.31199491, -0.6588304 , -0.81838223,  0.63823793],
                    [ 0.30303885, -0.58937555, -0.99182364,  0.7028515 ]]]

        # Predict output:
        prediction = lstm_model.predict(arguments)
        # Transform back to original values:
        prediction_copies = np.repeat(prediction, 4, axis=-1)
        result = lstm_scaler.inverse_transform(prediction_copies)[:, 3]

        return jsonify(str(result[0].astype(int)))


if __name__ == '__main__':
    app.run(debug=True)

