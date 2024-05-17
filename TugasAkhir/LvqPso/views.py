from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from .pso import PSO
from .lvq import LVQ

class App:
    # <--- Bagian preprocess Data --->
    # fungsi untuk membaca file data
    def preprocess_data(self, filename):
        df = pd.read_csv(filename)
        df.drop(columns="id", inplace=True)
        df = df[df['gender'] != 'Other']
        df['bmi'].fillna(df['bmi'].mean(), inplace = True)
        df['age'] = df['age'].astype(int)
        
        label_encoder = LabelEncoder()
        categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column])
        
        # mengambil fitur data
        data = df[df.columns[:-1]]
        label = df.iloc[:,-1:]
        
        # SMOTE Oversampling
        smote = SMOTE(random_state=42, k_neighbors=5)
        data, label = smote.fit_resample(data, label) 

        # UnderSampling
        # rus = RandomUnderSampler(random_state=42)
        # data, label = rus.fit_resample(data, label)
        
        data = data.values.round(3).tolist()
        label = label.values.tolist()
        labels = []
        for i in range(len(label)):
            for j in range(len(label[i])):
                labels.append(label[i][j])

        # split data
        data_train, data_test, label_train, label_test = train_test_split(data, label, test_size = 0.3)
        
        # Normalisasi
        scaler = MinMaxScaler()
        data_train = scaler.fit_transform(data_train)
        data_test = scaler.transform(data_test)
        data_train = np.round(data_train, 3)
        data_test = np.round(data_test, 3)
        
        labelTrain = np.array(label_train)
        label_train = labelTrain.flatten()
        labelTest = np.array(label_test)
        label_test = labelTest.flatten()

        total_row_train = len(data_train)
        total_row_test = len(data_test)
        total_col = len(df.axes[1])-1
        
        return (data_train, data_test, label_train, label_test, total_row_train, total_row_test, total_col)


    # <--- Bagian Training PSO-LVQ --->
    def train_psoLvq(self, data, label, row_data, column_data, maxIterasi, partikel, w, c1, c2, row_weight, epoch_lvq, lr, minimum_lr, pengurang_lr):
        weight_pso, iteration_pso = PSO.train_pso(PSO, data, label, row_data, maxIterasi, partikel, w, c1, c2)
        weight_psoLvq, epoch_psoLvq = LVQ.train_lvq(LVQ, data, row_data, column_data, weight_pso,  row_weight, label, epoch_lvq, lr, minimum_lr, pengurang_lr)
        return weight_psoLvq, iteration_pso


    # <--- Bagian Training Bobot PSO-LVQ dan LVQ --->
    def training(self, maxIterasi, partikel, w, c1, c2, epoch_lvq, lr, pengurang_lr, minimum_lr):
        data_train, data_test, label_train, label_test, row_data_train, row_data_test, column_data = self.preprocess_data(self, r"./static/data/stroke.csv")
        row_weight = 2
        
        # Training pso lvq
        weight_psoLvq, iteration_pso = self.train_psoLvq(self, data_train, label_train, row_data_train, column_data, maxIterasi, partikel, w, c1, c2, row_weight, epoch_lvq, lr, minimum_lr,pengurang_lr)
        
        # Inisialisasi bobot lvq
        weight, weight_label, data_train, label_train = LVQ.defineWeight(LVQ, data_train, label_train)
        row_data_train = len(data_train)
        
        # Train lvq murni
        weight_lvq, iteration_lvq = LVQ.train_lvq(LVQ, data_train, row_data_train, column_data, weight, row_weight, label_train, epoch_lvq, lr, minimum_lr, pengurang_lr)
        
        return (weight_psoLvq, weight_lvq, iteration_pso, iteration_lvq, data_test, label_test, row_data_test, column_data)


    # <--- Bagian Cek Akurasi PSO-LVQ dan LVQ --->
    def index(request):
        title = "Klasifikasi LVQ-PSO"
        context = {"title": title}
        if request.POST:
            # convert variabel
            epoch_lvq = int(request.POST.get("epoch_lvq"))
            lr = float(request.POST.get("learning_rate"))
            pengurang_lr = float(request.POST.get("pengurang_lr"))
            minimum_lr = float(request.POST.get("minimum_lr"))
            maxIterasi = int(request.POST.get("maxIterasi"))
            partikel = int(request.POST.get("partikel"))
            w = float(request.POST.get("koefisien_w"))
            c1 = float(request.POST.get("koefisien_c1"))
            c2 = float(request.POST.get("koefisien_c2"))

            acc_psoLvq = []
            acc_lvq = []
            iter_pso = []
            iter_lvq = []

            # training dan testing sesuai jumlah percobaan yang diinginkan
            for i in range(1):
                # bagian training bobot
                (weight_psoLvq, weight_lvq, iteration_pso, iteration_lvq, data, label, row_data, column_data) = App.training(App, maxIterasi, partikel, w, c1, c2, epoch_lvq, lr, pengurang_lr, minimum_lr)
                iter_pso.append(iteration_pso)
                iter_lvq.append(iteration_lvq)

                # bagian testing untuk akurasi
                # psoLvq
                (accuracy_psoLvq, not_disease_psoLvq, disease_psoLvq, classification_psoLvq, count_psoLvq) = LVQ.testing(LVQ, row_data, 2, column_data, data, weight_psoLvq, label)
                precisionScore_psoLvq = round(precision_score(label, classification_psoLvq)* 100, 3)
                recallScore_psoLvq = round(recall_score(label, classification_psoLvq)* 100, 3)
                f1Score_psoLvq = round(f1_score(label, classification_psoLvq)* 100, 3)

                acc_psoLvq.append(accuracy_psoLvq)
                precision_psoLvq = f"{precisionScore_psoLvq}"
                recall_psoLvq = f"{recallScore_psoLvq}"
                f1_psoLvq = f"{f1Score_psoLvq}"
                not_disease_psoLvq = f"{not_disease_psoLvq}"
                disease_psoLvq = f"{disease_psoLvq}"
                
                # lvq standar
                (accuracy_lvq, not_disease_lvq, disease_lvq, classification_lvq, count_lvq) = LVQ.testing(LVQ, row_data, 2, column_data, data, weight_lvq, label)
                precisionScore_lvq = round(precision_score(label, classification_lvq)* 100, 3)
                recallScore_lvq = round(recall_score(label, classification_lvq)* 100, 3)
                f1Score_lvq = round(f1_score(label, classification_lvq)* 100, 3)

                acc_lvq.append(accuracy_lvq)
                precision_lvq = f"{precisionScore_lvq}"
                recall_lvq = f"{recallScore_lvq}"
                f1_lvq = f"{f1Score_lvq}"
                not_disease_lvq = f"{not_disease_lvq}"
                disease_lvq = f"{disease_lvq}"
                
                classification_list = zip(label, classification_lvq, classification_psoLvq)

            print("acc pso lvq : ", acc_psoLvq)
            print("acc lvq : ", acc_lvq)
            print("precision PSO-LVQ:", precision_psoLvq)
            print("precision LVQ:", precision_lvq)
            print("recall PSO-LVQ:", recall_psoLvq)
            print("recall LVQ:", recall_lvq)
            print("f1-Score PSO-LVQ:", f1_psoLvq)
            print("f1-Score LVQ:", f1_lvq)
            print("iter pso : ", iter_pso)
            print("iter lvq : ", iter_lvq)
            
            context = {
                "title": title,
                "maxIterasi": maxIterasi,
                "partikel": partikel,
                "koefisien_w": w,
                "koefisien_c1": c1,
                "koefisien_c2": c2,
                "epoch_lvq": epoch_lvq,
                "learning_rate": lr,
                "pengurang_lr": pengurang_lr,
                "minimum_lr": minimum_lr,
                "accuracy_psoLvq": accuracy_psoLvq,
                "precision_psoLvq": precision_psoLvq,
                "recall_psoLvq": recall_psoLvq,
                "f1_psoLvq": f1_psoLvq,
                "not_disease_psoLvq": not_disease_psoLvq,
                "disease_psoLvq": disease_psoLvq,
                "accuracy_lvq": accuracy_lvq,
                "precision_lvq": precision_lvq,
                "recall_lvq": recall_lvq,
                "f1_lvq": f1_lvq,
                "not_disease_lvq": not_disease_lvq,
                "disease_lvq": disease_lvq,
                "classification_list": classification_list,
                "count_psoLvq": count_psoLvq,
                "count_lvq": count_lvq,
            }
        return render(request, "index.html", context)
