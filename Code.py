# Library Installation Import
import os
import sys
import subprocess
import pkg_resources

# List of Required Packages (Exclude Tkinter)
required_packages = ['pandas', 'scikit-learn']

# Function to Install Missing Packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"{package} is Installed.")

# Check if all Required Packages are Installed
def check_and_install_packages():
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]

    if missing_packages:
        print(f"Missing packages: {missing_packages} Installing")
        for package in missing_packages:
            install(package)
    else:
        print("All Necessary Packages are Already Installed!")

# Check and Install Necessary Packages
check_and_install_packages()

# Importing Required Libraries after Checking for Installation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tkinter import Tk, Label, Entry, Button, StringVar, OptionMenu, Frame, messagebox, PhotoImage
from tkinter.font import Font

# Loading the Dataset
file_path = 'Data_for_Farmers_Forest.csv'

if not os.path.exists(file_path):
    print(f"Error: The File '{file_path}' Does Not Exist. Please Check the Path and Try Again.")
    exit()

data = pd.read_csv(file_path)

# Map Month Names to Integers
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
data['Months'] = data['Months'].map(month_mapping)

# Combine Features for Modeling
data['Avg_pH'] = data[['pH_1', 'pH_2', 'pH_3']].mean(axis=1)
data['Avg_Soil'] = data[['Soil_1', 'Soil_2', 'Soil_3']].apply(lambda x: ','.join(x.unique()), axis=1)
data['Avg_Water'] = data[['Water_1', 'Water_2', 'Water_3']].mean(axis=1)

# Features and Targets
features = data[['Months', 'Temp', 'Avg_pH', 'Avg_Soil', 'Avg_Water']]
target_crop_1 = data['Crop_1']
target_crop_2 = data['Crop_2']
target_crop_3 = data['Crop_3']

# Define Possible Soil Types
soil_types = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']

# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Months', 'Temp', 'Avg_pH', 'Avg_Water']),
        ('soil', OneHotEncoder(categories=[soil_types], handle_unknown='ignore'), ['Avg_Soil'])
    ])

# Create and Fit Pipelines for Each Crop
def create_pipeline():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

# Fit the Preprocessor and Models
model_pipeline_crop_1 = create_pipeline().fit(features, target_crop_1)
model_pipeline_crop_2 = create_pipeline().fit(features, target_crop_2)
model_pipeline_crop_3 = create_pipeline().fit(features, target_crop_3)

# Special Crop Soil Preferences
crop_soil_preferences = {
    'Cotton': ['Black'],
    'Ragi': ['Red']
}

# Function to Predict the Best Crop Based on Input
def predict_best_crop_with_soil_dependency(month_int, temp, avg_pH, avg_soil, avg_water):
    # Prepare Input Data
    input_data = pd.DataFrame({
        'Months': [month_int],
        'Temp': [temp],
        'Avg_pH': [avg_pH],
        'Avg_Soil': [avg_soil],
        'Avg_Water': [avg_water]
    })

    # Transform Input Data
    input_data_transformed = preprocessor.transform(input_data)

    # Predict Probabilities for Each Crop
    prob_crop_1 = model_pipeline_crop_1.named_steps['classifier'].predict_proba(input_data_transformed)[0]
    prob_crop_2 = model_pipeline_crop_2.named_steps['classifier'].predict_proba(input_data_transformed)[0]
    prob_crop_3 = model_pipeline_crop_3.named_steps['classifier'].predict_proba(input_data_transformed)[0]

    # Confidence Factor Based on Environmental Conditions
    temp_score = (50 - abs(temp - 25)) / 50
    pH_score = (10 - abs(avg_pH - 7)) / 10
    
    # No Reduction If Water Input is Higher Than Required
    water_score = 1 if avg_water >= 1000 else avg_water / 1000
    soil_score = 1
    
    # Confidence Factor to Adjust Probabilities
    confidence_factor = 0.3 * temp_score + 0.3 * water_score + 0.2 * pH_score + 0.2 * soil_score
    prob_crop_1 = [p * confidence_factor for p in prob_crop_1]
    prob_crop_2 = [p * confidence_factor for p in prob_crop_2]
    prob_crop_3 = [p * confidence_factor for p in prob_crop_3]

    # Apply Additional Boost to Crops Based on Soil Preferences
    def apply_soil_preference(crop_name, avg_soil, probability):
        if crop_name in crop_soil_preferences and avg_soil in crop_soil_preferences[crop_name]:
            return min(probability * 1.3, 1.0)
        return probability

    # Aggregate Crops and Adjusted Probabilities into a List
    crops = [model_pipeline_crop_1.classes_, model_pipeline_crop_2.classes_, model_pipeline_crop_3.classes_]
    probabilities = [prob_crop_1, prob_crop_2, prob_crop_3]

    crop_list = []
    for crop_classes, crop_prob in zip(crops, probabilities):
        for i in range(len(crop_classes)):
            crop_name = crop_classes[i]
            prob_adjusted = apply_soil_preference(crop_name, avg_soil, crop_prob[i])
            crop_list.append((crop_name, prob_adjusted * 100))

    # Normalize Probabilities if > 100%
    max_prob = max(prob for _, prob in crop_list)
    if max_prob > 100:
        crop_list = [(crop_name, min(prob, 100)) for crop_name, prob in crop_list]

    # Sort Crops by Adjusted Probability
    crop_list.sort(key=lambda x: x[1], reverse=True)

    # Return the Best and Second-Best Crops
    best_crop, best_prob = crop_list[0]
    second_best_crop, second_best_prob = crop_list[1]

    return best_crop, best_prob, second_best_crop, second_best_prob

# GUI Class for Crop Prediction
class CropPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Farmers Forest (Crop Predictor App)")
        self.root.geometry("600x690") 
        self.root.resizable(False, False)
        self.root.configure(bg="#2e2e2e")
        self.icon_path = 'Farmers_Forest.png'
        self.icon_image = PhotoImage(file=self.icon_path)
        self.root.iconphoto(True, self.icon_image)

        # Center the Window
        self.window_width, self.window_height = 600, 690
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.position_top = int(self.screen_height / 2 - self.window_height / 2)
        self.position_right = int(self.screen_width / 2 - self.window_width / 2)
        self.root.geometry(f"{self.window_width}x{self.window_height}+{self.position_right}+{self.position_top}")

        # Fonts and Colors for Styling
        self.title_font = Font(family="Yu Gothic", size=16, weight="bold")
        self.label_font = Font(family="Yu Gothic", size=12)
        self.entry_font = Font(family="Yu Gothic", size=12)
        self.button_font = Font(family="Yu Gothic", size=14, weight="bold")
        self.result_font = Font(family="Yu Gothic", size=14, weight="bold")
        self.label_fg = "white"
        self.entry_bg = "#f0f0f0"
        self.entry_fg = "black"
        self.button_bg = "#0F8079"
        self.button_fg = "white"
        
        # Title Frame and Label
        self.title_frame = Frame(root, bg='#3b3b3b')
        self.title_frame.pack(pady=10, padx=10, fill='x')
        self.title_label = Label(self.title_frame, text="Farmers Forest - Crop Prediction", font=self.title_font, fg='white', bg='#3b3b3b')
        self.title_label.pack(pady=10)

        # Main Frame
        self.frame = Frame(root, bg='#2e2e2e')
        self.frame.pack(pady=10)

        # Month Dropdown
        self.label_month = Label(self.frame, text="Month:", font=self.label_font, bg="#2e2e2e", fg=self.label_fg)
        self.label_month.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.month_var = StringVar(self.root)
        self.month_var.set("Select Month")
        self.months_list = list(month_mapping.keys())
        self.dropdown_month = OptionMenu(self.frame, self.month_var, *self.months_list)
        self.dropdown_month.config(font=self.entry_font)
        self.dropdown_month.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Temperature Entry
        self.label_temp = Label(self.frame, text="Temperature (C):", font=self.label_font, bg="#2e2e2e", fg=self.label_fg)
        self.label_temp.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.entry_temp = Entry(self.frame, font=self.entry_font, bg=self.entry_bg, fg=self.entry_fg)
        self.entry_temp.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # pH Entry
        self.label_pH = Label(self.frame, text="Soil pH Level:", font=self.label_font, bg="#2e2e2e", fg=self.label_fg)
        self.label_pH.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.entry_pH = Entry(self.frame, font=self.entry_font, bg=self.entry_bg, fg=self.entry_fg)
        self.entry_pH.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Soil Type Dropdown
        self.label_soil = Label(self.frame, text="Soil Type:", font=self.label_font, bg="#2e2e2e", fg=self.label_fg)
        self.label_soil.grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.soil_var = StringVar(self.root)
        self.soil_var.set("Select Soil")
        self.dropdown_soil = OptionMenu(self.frame, self.soil_var, *soil_types)
        self.dropdown_soil.config(font=self.entry_font)
        self.dropdown_soil.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Water Entry
        self.label_water = Label(self.frame, text="Water Availability (mm):", font=self.label_font, bg="#2e2e2e", fg=self.label_fg)
        self.label_water.grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.entry_water = Entry(self.frame, font=self.entry_font, bg=self.entry_bg, fg=self.entry_fg)
        self.entry_water.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Predict and Exit Buttons
        self.button_predict = Button(self.frame, text="Predict Best Crop", command=self.predict_crop, font=self.button_font, bg=self.button_bg, fg=self.button_fg)
        self.button_predict.grid(row=6, column=0, columnspan=2, pady=10)

        self.button_exit = Button(self.frame, text="Exit", command=self.root.quit, font=self.button_font, bg=self.button_bg, fg=self.button_fg)
        self.button_exit.grid(row=7, column=0, columnspan=2, pady=10)

        # Result Labels
        self.label_best_crop = Label(self.frame, font=self.result_font, bg="#2e2e2e")
        self.label_best_crop.grid(row=8, column=0, columnspan=2, pady=5)
        self.label_second_best_crop = Label(self.frame, font=self.result_font, bg="#2e2e2e")
        self.label_second_best_crop.grid(row=9, column=0, columnspan=2, pady=5)

    def validate_inputs(self, month, temp, pH, water):
        # Validation for Missing Fields
        if not month or not temp or not pH or not water:
            messagebox.showerror("Error", "Please Fill All Fields.")
            return False
        
        # Limits Validation
        if not 0 <= temp <= 50:
            messagebox.showerror("Error", "Temperature Must be Between 0 and 50°C.")
            return False
        if not 4 <= pH <= 9:
            messagebox.showerror("Error", "pH Value Must be Between 4 and 9.")
            return False
        if not 0 <= water <= 300:
            messagebox.showerror("Error", "Water Availability Must be Between 0 and 300 mm.")
            return False
        return True

    def predict_crop(self):
        month = self.month_var.get()
        temp = self.entry_temp.get()
        pH = self.entry_pH.get()
        soil = self.soil_var.get()
        water = self.entry_water.get()

        if not temp or not pH or not water or month == "Select Month" or soil == "Select Soil":
            messagebox.showerror("Error", "Please Provide All Required Inputs.")
            return
        
        # Convert Input Values
        try:
            month_int = month_mapping[month]
            temp = float(temp)
            pH = float(pH)
            water = float(water)
        except ValueError:
            messagebox.showerror("Error", "Invalid Input. Please Enter Numerical Values for Temperature, pH, and Water.")
            return
        
        # Validate Input Ranges
        if not self.validate_inputs(month, temp, pH, water):
            return
        best_crop, best_prob, second_best_crop, second_best_prob = predict_best_crop_with_soil_dependency(month_int, temp, pH, soil, water)
        
        # Display Results with Color-Coded Confidence Scores
        best_color = "red" if best_prob < 40 else "green"
        second_best_color = "red" if second_best_prob < 40 else "green"

        self.label_best_crop.config(text=f"Best Crop: {best_crop} ({best_prob:.2f}%)", fg=best_color)
        self.label_second_best_crop.config(text=f"Second Best Crop: {second_best_crop} ({second_best_prob:.2f}%)", fg=second_best_color)

# Main Application
if __name__ == "__main__":
    root = Tk()
    app = CropPredictorApp(root)
    root.mainloop()
