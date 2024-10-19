from flask import Flask, request, jsonify
import joblib
import logging
import pandas as pd

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load the pre-trained classifier and label encoders
with open('label_encoder_age.sav', 'rb') as f:
    label_encoder_age = joblib.load(f)

with open('label_encoder_status.sav', 'rb') as f:
    label_encoder_status = joblib.load(f)

with open('nutritionmodel.sav', 'rb') as f:
    clf = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data.get('age')
    nutritional_status = data.get('nutritional_status')
    
    # Create a DataFrame from the input data
    new_data = pd.DataFrame({
        'AGE': [age],
        'NUTRITIONAL STATUS': [nutritional_status]
    })

    # Encode the features
    new_data['AGE_ENCODED'] = label_encoder_age.transform(new_data['AGE'])
    new_data['STATUS_ENCODED'] = label_encoder_status.transform(new_data['NUTRITIONAL STATUS'])

    # Select features for prediction
    X_new = new_data[['AGE_ENCODED', 'STATUS_ENCODED']]

    # Make predictions
    predictions = clf.predict(X_new)

    # Add predictions to the DataFrame
    new_data['NUTRITIONAL TIPS'] = predictions

    # Prepare the response
    response = new_data[['AGE', 'NUTRITIONAL STATUS', 'NUTRITIONAL TIPS']].iloc[0].to_dict()
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
