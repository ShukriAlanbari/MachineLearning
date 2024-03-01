import tkinter as tk
from joblib import load

# Load the model and converter
final_poly_convertor = load("insurance_converter_cv_fin_2024_01_30.joblib")
loaded_model = load("insurance_ridge_cv_fin_2024_01_30.joblib")

def model_func(age, sex, bmi, children, smoker, region_southwest, region_southeast, region_northwest, region_northeast):
    # Convert categorical variables to numerical format using one-hot encoding
    sex_numeric = int(sex)  # 0 for female and 1 for male
    smoker_numeric = int(smoker)  # 0 for non-smoker and 1 for smoker

    # Create a list of features in the same order as model expects
    features = [
        float(age), sex_numeric, float(bmi), float(children),smoker_numeric,
        region_southwest, region_southeast, region_northwest, region_northeast
    ]

    # Expand the feature vector to include polynomial features
    features_poly = final_poly_convertor.transform([features])

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(features_poly)

    return prediction[0]


# Tkinter GUI code
app = tk.Tk()
app.title("Insurance RidgeCV Model")
app.geometry("400x300")

label1 = tk.Label(app, text="Age:")
label1.grid(row=0, column=0)
entry1 = tk.Entry(app)
entry1.grid(row=0, column=1)

label2 = tk.Label(app, text="Sex:")
label2.grid(row=1, column=0)
entry2 = tk.Entry(app)
entry2.grid(row=1, column=1)

label3 = tk.Label(app, text="BMI:")
label3.grid(row=2, column=0)
entry3 = tk.Entry(app)
entry3.grid(row=2, column=1)

label4 = tk.Label(app, text="Children:")
label4.grid(row=3, column=0)
entry4 = tk.Entry(app)
entry4.grid(row=3, column=1)

label5 = tk.Label(app, text="Smoker:")
label5.grid(row=4, column=0)
entry5 = tk.Entry(app)
entry5.grid(row=4, column=1)

label6 = tk.Label(app, text="Region:")
label6.grid(row=5, column=0)
entry6 = tk.Entry(app)
entry6.grid(row=5, column=1)

# Define result_label as a global variable
result_label = tk.Label(app, text="", fg="red")
result_label.grid(row=7, column=0, columnspan=2)

# Define execute function before it's used
def execute():
    global result_label  # Declare result_label as global
    age = entry1.get()
    sex = entry2.get()
    bmi = entry3.get()
    children = entry4.get()
    smoker = entry5.get()
    region_southwest = 1 if entry6.get().lower() == 'southwest' else 0
    region_southeast = 1 if entry6.get().lower() == 'southeast' else 0
    region_northwest = 1 if entry6.get().lower() == 'northwest' else 0
    region_northeast = 1 if entry6.get().lower() == 'northeast' else 0

    result = model_func(age, sex, bmi, children, smoker, region_southwest, region_southeast, region_northwest, region_northeast)
    result_label.config(text=f"Result: {result}")

button = tk.Button(app, text="Execute", command=execute)
button.grid(row=6, column=0, columnspan=2)

app.mainloop()
