#Importing the Libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

with open('model.pkl','rb') as file:
    model = pickle.load(file)

st.title("💻 Laptop Price Prediction 💻")

#Gathering the Data
all_brands = ['HP', 'Acer', 'Lenovo', 'Apple', 'Dell', 'Asus', 'Samsung',
       'Ultimus', 'Primebook', 'MSI', 'Infinix', 'Wings', 'Honor',
       'Zebronics', 'Xiaomi', 'iBall', 'Chuwi', 'Realme', 'Avita',
       'Walker', 'Huawei', 'Tecno', 'Gigabyte', 'Vaio', 'Microsoft',
       'Fujitsu', 'LG', 'Ninkear', 'Razer', 'AXL']

all_processors = ['5th Gen AMD Ryzen 5 5600H', '12th Gen Intel Core i3 1215U',
       '11th Gen Intel Core i3 1115G4', '12th Gen Intel Core i5 1240P',
       'Apple M1', '13th Gen Intel Core i5 13420H',
       '12th Gen Intel Core i5 12500H', '12th Gen Intel Core i7 1255U',
       'Intel Celeron  N4020', 'MediaTek MTK8788',
       '7th Gen AMD Ryzen 3 7320U', '11th Gen Intel Core i5 11400H ',
       '13th Gen Intel Core i9 13900H', '12th Gen Intel Core i5 12450H',
       '11th Gen Intel Core i5 11300H', 'Apple M2',
       '11th Gen Intel Core i5 1135G7 ', '5th Gen AMD Ryzen 7  5800H',
       '5th Gen AMD Ryzen 5 5500U', '3rd Gen AMD Athlon 3050U',
       'Intel Core i3 N305', '13th Gen Intel Core i7 1355U',
       '6th Gen AMD Ryzen 5 6600H', '13th Gen Intel Core i9 13900HX',
       '12th Gen Intel Core i7 12650H', '13th Gen Intel Core i5 1340P',
       '12th Gen Intel Core i5 1235U ', '13th Gen Intel Core i5 13450HX',
       '13th Gen Intel Core i9 13980HX', '11th Gen Intel Core i5 1135G7',
       '5th Gen AMD Ryzen 3 5300U', '12th Gen Intel Core i7 12700H',
       '5th Gen AMD Ryzen 5 5625U', '11th Gen Intel Core i5 1155G7',
       '7th Gen AMD Ryzen 7 7730U', '12th Gen Intel Core i5 1235U',
       '11th Gen Intel Core i7 11800H', '12th Gen Intel Core i7 12700H ',
       '13th Gen Intel Core i7 13620H', '13th Gen Intel Core i7 1360P',
       '11th Gen Intel Celeron N5100', '5th Gen AMD Ryzen 3 5425U',
       '10th Gen Intel Core i7 10750H', '7th Gen AMD Ryzen 7 7840HS',
       '13th Gen Intel Core i5 13500H', '7th Gen AMD Ryzen 5 7535HS',
       '11th Gen Intel Core i5 11260H', '11th Gen Intel Core i5  1135G7',
       '6th Gen AMD Ryzen 7  6800H', 'AMD Athlon Pro 3045B',
       '5th Gen AMD Ryzen 5  5500U', '11th Gen Intel Core i5 11300H ',
       'Intel Atom Quad Core Z3735F', '12th Gen Intel Core i3 1220P',
       '6th Gen AMD Ryzen 7 6800H', '7th Gen Amd Ryzen 5 7535HS',
       '5th Gen AMD Ryzen 5  5600H', '10th Gen Intel Core i3 1005G1',
       '5th Gen AMD Ryzen 7   5700U', '13th Gen Intel Core i5 1340p',
       '11th Gen Intel Core i7 1165G7', '11th Gen Intel Core i3  1115G4',
       '13th Gen Intel Core i3 1315U', '13th Gen Intel Core i5 1335U',
       '5th Gen AMD Ryzen 5  5600H ', 'Intel Celeron  N4500',
       'AMD Athlon 7120U', '7th Gen AMD Ryzen 7 7735U',
       '13th Gen Intel Core i3 1305U', '11th Gen Intel Core i5 11320H',
       '5th Gen AMD Ryzen 5 5600HS', '12th Gen Intel Core i5 12500H ',
       '5th Gen AMD Ryzen 7 5800H', '7th Gen AMD Ryzen 7 7735HS',
       '13th Gen Intel Core i7 13700HX', '7th Gen AMD Ryzen 9  7945HX ',
       '7th Gen AMD Ryzen 5 7530U', '3rd Gen AMD Ryzen 3  3250U',
       '5th Gen AMD Ryzen 5  5500U ', '6th Gen AMD Ryzen 7 6800H ',
       'Apple M2 Apple M2 Chip', '10th Gen Intel Core i5 10210U ',
       'Apple M1 Apple M1 Chip', '3rd Gen AMD Ryzen 5  3500U',
       '9th Gen Intel Core i9', 'AMD Athlon Silver 7120U',
       '10th Gen Intel Core i5 1035G1', '13th Gen Intel Core i7 13700H',
       '7th Gen AMD Ryzen 9  7940HS', 'Intel Celeron N4500',
       '5th Gen AMD Ryzen 7 5800HS', '11th Gen Intel Core i3 1125G4',
       '11th Gen intel Core i3 1115G4', '12th Gen Intel Core i7 12650H ',
       '7th Gen AMD Ryzen 3 7330U', '7th Gen Amd Ryzen 5 7520U',
       '7th Gen AMD Ryzen 5 7520U', '13th Gen Intel Core i7 13650HX',
       '11th Gen Intel Core i5 11400H', '13th Gen Intel Core i5 13500HX',
       '11th Gen Intel Core i3 1115G4 ',
       '3rd Gen AMD Athlon Silver 3050U', 'Apple M2 Max M2 Max',
       'Apple M2 Pro M2 Pro', '5th Gen AMD Ryzen 3  5425U',
       'AMD Ryzen 3 7320U', '12th Gen Intel Core i5 1230U',
       '11th Gen Intel Core i7 1185G7', '5th Gen AMD Ryzen 7 5700U',
       '12th Gen Intel Core i5 12450H ', '6th Gen AMD Ryzen 9  6900HX',
       '5th Gen AMD Ryzen 7  5825U', '10th Gen Intel Core i5 10300H',
       '5th Gen AMD Ryzen 5 5625U ', '5th Gen AMD Ryzen 3  5300U',
       'Apple M1 Pro M1 Pro', 'Apple M1 Max M1 Max',
       '10th Gen Intel Core i7 10870H', '3rd Gen AMD Ryzen 5  3580U',
       '3rd Gen AMD Ryzen 5 3500U ', '8th Gen Intel Core i5 8265U',
       '8th Gen Intel Core i5 8250U', 'Intel',
       '12th Gen Intel Core i7 1265U', '12th Gen Intel Core i7 1260P',
       '5th Gen AMD Ryzen 5  5625U', 'Intel Celeron  N4020 ',
       '12th Gen Intel Core i7 1260P ', 'Intel Pentium Silver  N6000',
       '10th Gen Intel Core i3 10110U ', '5th Gen AMD Ryzen 7 5800H ',
       '10th Gen Intel Core i3', '10th Gen Intel Core i7',
       '8th Gen Intel Core i7 8550U', '9th Gen intel Core i7 9750H',
       '8th Gen Intel Core i9 8950HK', '5th Gen AMD Ryzen 7  5700U',
       '7th Gen AMD Ryzen 5 7520U ', '7th Gen AMD Ryzen 7  7730U',
       'Intel   N4500', '5th Gen AMD Ryzen 5 5500H',
       '7th Gen Amd Ryzen 7 7840H', '6th Gen AMD Ryzen 7 6800HS ',
       '13th Gen Intel Core i9 13950HX', 'Intel Celeron Dual Core N4500',
       '4th Gen AMD Ryzen 7 PRO 4750U', '5th Gen AMD Ryzen 5  5500H',
       '13th Gen Intel Core i7 1355U ', '7th Gen AMD Ryzen 5  7530U',
       'Intel Celeron N4020', '12th Gen Intel Core i7 1250U',
       '13th Gen Intel Core i5 1334U', '7th Gen AMD Ryzen 9 7940HS',
       '13th Gen Intel Core i7 1360p', '12th Gen Intel Core i9 12900H',
       '10th Gen Intel Core i5 1035G4', '10th Gen Intel Core i5 10210U',
       '13th Gen Intel Core i7 1365U', 'intel Celeron  N4020',
       '11th Gen Intel Core i7 1195G7', '3rd Gen AMD Athlon  3050U',
       '5th Gen AMD Ryzen 7 5825U', '7th Gen Amd Ryzen 7 7745HX',
       '10th Gen Intel Core i5 10310U', 'Intel Pentium Silver   N6000 ',
       '6th Gen AMD Ryzen 7  7735HS', '4th Gen Intel Celeron N4020',
       '11th Gen Intel Core i7 1165G7 ',
       '13th Gen Intel Core i5 1335U', '3rd Gen AMD Ryzen 5 3500U',
       '10th Gen Intel Core i7 10750H ', '7th Gen Amd Ryzen 5 7530U',
       '7th Gen Amd Ryzen 9 7940HS', '7th Gen AMD Ryzen 7  7735HS',
       '13th Gen Intel Core i5 13500H ', '7th Gen AMD Ryzen 7 7745HX',
       '5th Gen AMD Ryzen 5  5500u', 'Intel Core i7',
       '13th Gen Intel 7 13700HX ', 'Intel Pentium Silver N6000',
       '5th Gen AMD Ryzen 7 5800U', '7th Gen AMD Ryzen 7040 Series 040',
       '13th Gen Intel Core i9 13900HX ',
       '13th Gen Intel Core i3 1315U']

all_cpu_comb = ['Hexa Core, 12 Threads', 'Hexa Core (2P + 4E), 8 Threads',
       'Dual Core, 4 Threads', '12 Cores (4P + 8E), 16 Threads',
       'Octa Core (4P + 4E)', 'Octa Core (4P + 4E), 12 Threads',
       '10 Cores (2P + 8E), 12 Threads', 'Dual Core, 2 Threads',
       'Octa Core', 'Quad Core, 8 Threads',
       '14 Cores (6P + 8E), 20 Threads', 'Octa Core, 16 Threads',
       'Octa Core, 8 Threads', '24 Cores (8P + 16E), 32 Threads',
       '10 Cores (6P + 4E), 16 Threads', '14 Cores, 20 Threads',
       'Quad Core, 4 Threads', 'Quad Core',
       '5 Cores (1P + 4E), 6 Threads', '16 Cores (8P + 8E), 24 Threads',
       '16 Cores, 32 Threads', '14 Cores (6P + 8E)', '12 Cores',
       '10 Cores, 12 Threads', 'Octa Core, 12 Threads',
       '10 Cores (8P + 2E)', '10 Cores', '24 Cores (8P + 16E)',
       '20 Threads']

all_gpus = ['4GB AMD Radeon RX 6500M', 'Intel UHD Graphics',
       'Intel Iris Xe Graphics', 'Intel Integrated Iris Xe',
       'Apple M1 Integrated Graphics', '6GB NVIDIA GeForce RTX 4050',
       'Intel Iris Xe', 'Intel Integrated UHD',
       'Intel Integrated UHD Graphics', '4GB NVIDIA GeForce RTX 3050',
       'ARM Mali G72', 'AMD Radeon Graphics',
       '4GB NVIDIA GeForce RTX 2050', '4GB NVIDIA GeForce GTX 1650',
       'AMD Radeon Vega 7', '8-Core GPU', 'AMD Radeon AMD',
       'AMD Integrated', '8GB NVIDIA GeForce RTX 4070',
       '4GB AMD Radeon RX 6500M Graphics', 'AMD Graphics',
       '8GB NVIDIA GeForce RTX 3070 Ti', 'Intel Integrated Integrated',
       'Intel Graphics', '6GB NVIDIA GeForce RTX 3050',
       'Intel UHD Graphics ', '16GB NVIDIA GeForce RTX 4090',
       'AMD Radeon Radeon Graphics', '8GB NVIDIA GeForce RTX 4060',
       'Intel Integrated Intel UHD Graphics',
       '4GB NVIDIA GeForce RTX 3050 Ti', 'Intel Integrated Intel UHD',
       '4GB NVIDIA GeForce GTX 3050', 'AMD Radeon Vega 7 Graphics',
       '4GB NVIDIA ', 'Intel Integrated', 'Intel Iris XE Graphics ',
       'GB NVIDIA GeForce RTX 2050', '8GB NVIDIA GeForce RTX 3070 Ti ',
       '10-Core GPU', '4GB NVIDIA GeForce RTX 3050 Graphics',
       '8GB AMD Radeon RX 6650M', 'Intel HD Graphics',
       'Intel UHD Graphics 600', 'Integrated Intel Iris Xe Graphics',
       'AMD Radeon Radeon', 'Intel Integrated Intel Iris Xe Graphics',
       'AMD Radeon Graphiics', 'NVIDIA GeForce RTX 4070',
       'AMD Radeon 610M Graphics', '4GB NVIDIA GTX 1650', 'AMD Radeon',
       '4GB NVIDIA Geforce RTX 3050', 'Intel Graphics ',
       'Intel Iris XE Graphics', 'Intel HD', 'AMD Radeon Vega 8',
       '4GB Radeon Pro 5500M', 'AMD Radeon 610M',
       '4GB NVIDIA GeForce RTX RTX 3050', 'AMD Radeon ',
       '4GB AMD Radeon AMD Radeon RX 6500M',
       '4GB NVIDIA GeForce GTX 2050', 'Integrated Intel UHD Graphics',
       '4GB NVIDIA GeForce RTX2050', 'Intel Iris X Graphics',
       '4GB NVIDIA GeForce RTX 2050 ', '4GB NVIDIA GeForce RTX 3050 ',
       'AMD Radeon Vega 8 Graphics', '4GB Nvidia GeForce RTX 2050',
       'Integrated Intel iris Xe Graphics', '4GB AMD Radeon RX6500M',
       '12GB NVIDIA GeForce RTX 4080', 'Intel Iris Xe Graphic',
       '30-core GPU', '19-core GPU', ' Intel Iris Xe Graphics',
       '6GB NVIDIA GeForce RTX 3060', '8GB NVIDIA GeForce RTX 3070Ti',
       '4GB NVIDIA GeForce RTX A1000', '16GB NVIDIA GeForce RTX 3080 Ti',
       'Intel Iris Xe Graphics ', '4GB NVIDIA Geforce GTX1650 Max Q',
       '16-core GPU', '32-core GPU', '2GB NVIDIA GeForce MX350',
       'AMD Radeon Vega 6', 'AMD Radeon Vega 9',
       '2GB NVIDIA GeForce MX250', 'Integrated UHD Graphics 620',
       'Intel Integrated UHD 620', 'Intel Iris Plus Graphics 655',
       '4GB Intel Arc A350M Graphics', 'Intel Iris Plus',
       '4GB Geforce MX130', 'Intel UHD Graphics 630',
       '4GB NVIDIA Geforce GTX 1050Ti', 'Integrated AMD Radeon Graphics',
       '6GB NVIDIA Geforce RTX 4050', '16GB NVIDIA GeForce RTX 4090 ',
       'AMD Radeon 780M Graphics', '12GB NVIDIA Geforce RTX 4080',
       '8GB NVIDIA RTX 3000 Ada', '4GB AMD Radeon RX6550M',
       '4GB  NVIDIA GeForce RTX 2050', 'Intel Integrated Graphics',
       'Intel Integrated Graphics ', '8GB NVIDIA GeForce RTX A2000 Ada',
       'Intel Integrated Iris', 'Intel Integrated UHD 600',
       'Intel Integrated Integrated Intel® UHD Graphics',
       'NVIDIA Geforce RTX', 'AMD Integrated SoC',
       '4GB NVIDIA GeForce RTX 3050 GPU', '2GB NVIDIA GeForce MX450',
       '6GB NVIDIA RTX 4050', '8GB NVIDIA GEFORCE RTX 4060',
       '4GB NVIDIA RTX A500 ', '12GB NVIDIA GeForce RTX 3080 Ti',
       '6GB NVIDIA GeForce RTX 3050 ', 'AMD Radeon 7 Graphics',
       'Iris Xe Graphics', 'AMD Radeon 680M', '4GB NVIDIA Quadro T600',
       'Intel UHD', '8GB Nvidia GeForce RTX 4060',
       '6GB Nvidia GeForce RTX 4050', '2GB NVIDIA GeForce MX570',
       '8GB NVIDIA Geforce RTX 4060', '4GB Nvidia RTX 3050',
       '4GB NVIDIA GeForce GTX 1650 ', '6GB NVIDIA GEFORCE RTX 4050',
       'Intel UHD Integrated', '4GB AMD Radeon RX 5600M ']

all_os = ['Windows 11 OS', 'Mac OS', 'Android 11 OS', 'DOS OS',
       'Windows 10 OS', 'Windows 10  OS', 'Chrome OS', 'Windows OS',
       'Ubuntu OS', 'Mac Catalina OS', 'DOS 3.0 OS', 'Windows 11  OS',
       'Mac High Sierra OS', 'Mac 10.15.3 OS']

ramopt = [1,2,4,6,8,12,16,32,64,128]
ramtypeopt = ['DDR4', 'LPDDR5', 'DDR5', 'LPDDR4', 'LPDDR5X', 'LPDDR4X', 'DDR3',
       'LPDDR4x', 'Unified', 'DDR4-', 'LPDDR5x', 'DDR']

# UI components
brand = st.selectbox("Select Brand", all_brands)
spec_rating = st.number_input("Enter Spec Rating", min_value=0, max_value=100, value=40)
processor = st.selectbox("Select Processor", all_processors)
CPU = st.selectbox("Enter CPU Cores + Threads",all_cpu_comb)
Ram = st.selectbox("Enter RAM (GB)",ramopt)
Ram_type = st.selectbox("Enter RAM Type", ramtypeopt)
ROM = st.number_input("Enter Storage (GB)", min_value=1, value=256)
ROM_type = st.radio("Select Storage Type", ["SSD", "HDD"])
GPU = st.selectbox("Select GPU", all_gpus)
display_size = st.text_input("Enter Display Size", "15.6")
resolution_width = st.text_input("Enter Resolution Width", "1920")
resolution_height = st.text_input("Enter Resolution Height", "1080")
OS = st.selectbox("Select OS", all_os)
warranty = st.text_input("Warranty Period", "1.0")


def process_input(raw_data, training_columns):
    # Ensure all training columns are present in the input dataframe
    for col in training_columns:
        if col not in raw_data:
            raw_data[col] = 0  # Set a default value or handle as needed

    # Ensure consistent encoding logic (e.g., one-hot encoding)
    processed_data = pd.get_dummies(raw_data, columns=training_columns)

    # Reorder columns to match the order during training
    processed_data = processed_data[training_columns]

    return processed_data

#exporting the data
if st.button("Predict"):
       user_data = {
              'brand': [brand],
              'spec_rating': [spec_rating],
              'processor': [processor],
              'CPU': [CPU],
              'Ram': [Ram],
              'Ram_type': [Ram_type],
              'ROM': [ROM],
              'ROM_type': [ROM_type],
              'GPU': [GPU],
              'display_size': [display_size],
              'resolution_width': [resolution_width],
              'resolution_height': [resolution_height],
              'OS': [OS],
              'warranty': [warranty]
       }
       # Create a DataFrame from user inputs
       user_df = pd.DataFrame(user_data)
       st.markdown(
              """
              <style>
                     body {
                            background-color: #f0f0f0;  /* Set your desired background color */
                     }
                     .css-class-name {
                     /* Add your custom CSS styles here */
                     color: #333;
                     font-size: 18px;
                     }
              </style>
              """,
              unsafe_allow_html=True
       )


# Example usage
       enc = LabelEncoder()
       categorical_columns = ['brand', 'processor', 'CPU', 'Ram_type', 'ROM_type', 'GPU', 'OS']
       for column in categorical_columns:
              user_df[column] = enc.fit_transform(user_df[column])
       prediction = model.predict(user_df)
       st.write(prediction)
