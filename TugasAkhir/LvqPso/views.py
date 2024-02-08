from django.shortcuts import render
import pandas as pd
import numpy as np
import math
from random import random
from itertools import islice
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler

# <--- Bagian Read Data --->
# fungsi untuk membaca file data
def read_data(filename):
    # mebaca file csv
    df = pd.read_csv(filename)
    # mengambil fitur data
    data = df[df.columns[:-1]]
    data = data.values.round(3).tolist()
    label = df.iloc[:, -1:]
    label = label.values.tolist()
    labels = []
    for i in range(len(label)):
        for j in range(len(label[i])):
            labels.append(label[i][j])
    
    # SMOTE Oversampling
    smote = SMOTE(random_state=42, k_neighbors=5)
    X, y = smote.fit_resample(data, label) 
    
    # UnderSampling
    # rus = RandomUnderSampler(random_state=42)
    # X, y = rus.fit_resample(data, label)
    
    data_train, data_test, label_train, label_test = train_test_split(X, y, test_size=0.2)
    
    # Normalisasi
    scaler = MinMaxScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    
    labelTrain = np.array(label_train)
    label_train = labelTrain.flatten()
    labelTest = np.array(label_test)
    label_test = labelTest.flatten()

    total_row_train = len(data_train)
    total_row_test = len(data_test)
    total_col = len(df.axes[1]) - 1
    print(total_row_train)
    return (data_train, data_test, label_train, label_test, total_row_train, total_row_test, total_col)
# <--- Akhir Bagian Read Data --->

# fungsi inisialisasi weight lvq random
def defineWeight(data_train, label_train):
    data_train = np.array(data_train)
    label_train = np.array(label_train)
    weight_label, label_index = np.unique(label_train, True)
    weight = data_train[label_index].astype(np.float)
    dataTrain = np.delete(data_train, label_index, axis=0)
    labelTrain = np.delete(label_train, label_index, axis=0)
    dataTrain = dataTrain.tolist()
    labelTrain = labelTrain.tolist()
    weight = weight.tolist()
    return weight, weight_label, dataTrain, labelTrain

# <--- Bagian Metode PSO --->
# fungsi untuk inisialisasi kecepatan awal = 0
def initV(size):
    column = 20
    velocity = [[0] * column] * size
    for i in range(size):
        for j in range(column):
            velocity[i][j] = 0
    return velocity

# fungsi untuk inisialisasi partikel awal secara random
def initPartikel(size):
    column = 20
    position = []
    for i in range(size):
        row = []
        for j in range(column):
            row.append(round(random(), 3))
        position.append(row)
    return position

# fungsi optimasi mencari nilai cost data
def costData(row_data, row_particle, data, particle, label):
    sqrt_value = []
    # rumus menghitung jarak minimal
    for i in range(row_particle):
        temp = []
        for j in range(row_data):
            calculate0, calculate1 = 0, 0
            # untuk kelas 0 (index partikel 1 - 10)
            for k in range(0, 10):
                calculate0 += pow(data[j][k] - particle[i][k], 2)
            # untuk kelas 1 (index partikel 11 - 20)
            for k in range(10, 20):
                calculate1 += pow(data[j][k - 10] - particle[i][k], 2)
            temp.append([math.sqrt(calculate0), math.sqrt(calculate1)])
        sqrt_value.append(temp)
    cost_value = []
    mins_value = []
    mins = 0
    # menyimpan jarak kelas minimal kelas 0/1
    for i in range(row_particle):
        minDistance = []
        for j in range(row_data):
            if sqrt_value[i][j][0] <= sqrt_value[i][j][1]:
                mins = 0
            else:
                mins = 1
            minDistance.append(mins)
        mins_value.append(minDistance)
    # check perbedaan hasil jarak minimal(cost) dengan target data(kelas sebenarnya)
    for i in range(row_particle):
        cost, value = 0, 0
        for j in range(row_data):
            if mins_value[i][j] == label[j]:
                value = 0
            else:
                value = 1
            cost += value
        cost_value.append(cost)
    return cost_value

# fungsi menghitung nilai fitness
def calculateFitness(cost_value, row_data):
    fitness_value = []
    for i in range(len(cost_value)):
        fitness = 0
        fitness = (row_data - cost_value[i]) / row_data
        fitness_value.append(fitness)
    return fitness_value

# fungsi inisialisasi Pbest
def initialPBest(particle):
    pBest = particle
    return pBest

# fungsi update Pbest
def updatePBest(oldFitness, newFitness, oldPBest, newPosition):
    new_pbest = []
    for i in range(len(oldFitness)):
        if oldFitness[i] > newFitness[i]:
            new_pbest.append(oldPBest[i])
        elif oldFitness[i] == newFitness[i]:
            new_pbest.append(oldPBest[i])
        else:
            new_pbest.append(newPosition[i])
    return new_pbest

# fungsi set Gbest
def setGBest(pbest, fitness):
    index = fitness.index(max(fitness))
    return pbest[index], fitness[index]

# fungsi update kecepatan partikel
def updateVelocity(w, c1, c2, particle, pbest, gbest, velocity):
    newVelocity = []
    for i in range(len(velocity)):
        temp = []
        for j in range(len(velocity[i])):
            calculate = round(w * velocity[i][j] 
                        + c1 * round(random(), 3) * (pbest[i][j] - particle[i][j]) 
                        + c2 * round(random(), 3) * (gbest[j] - particle[i][j]), 3)
            temp.append(calculate)
        newVelocity.append(temp)
    return newVelocity

# fungsi update posisi partikel
def updatePosition(particle, newVelocity):
    newParticle = []
    for i in range(len(particle)):
        temp = []
        for j in range(len(particle[i])):
            temp.append(round(particle[i][j] + newVelocity[i][j], 3))
        newParticle.append(temp)
    return newParticle

# fungsi untuk convert gbest menjadi 2 list untuk masing" kelas
def convert(lst, var_lst):
    it = iter(lst)
    return [list(islice(it, i)) for i in var_lst]
# <--- Akhir Bagian Metode PSO --->


# <--- Bagian Metode LVQ --->
# fungsi untuk menghitung jarak euclidean
def euclideanLVQ(row_data, row_weight, column_data, data, weight, label, alpha):
    updatedWeight = weight
    for i in range(row_data):
        sqrt_value = []
        for j in range(row_weight):
            calculate = 0
            for k in range(column_data):
                calculate += pow(data[i][k] - weight[j][k], 2)
            sqrt_value.append(math.sqrt(calculate))
        if sqrt_value[0] <= sqrt_value[1]:
            mins = 0
        else:
            mins = 1
        if mins == label[i]:
            for k in range(column_data):
                updatedWeight[mins][k] = round(
                    weight[mins][k] + (alpha * (data[i][k] - weight[mins][k])), 6)
        else:
            for k in range(column_data):
                updatedWeight[mins][k] = round(
                    weight[mins][k] - (alpha * (data[i][k] - weight[mins][k])), 6)
        weight = updatedWeight
    return weight

# fungsi update Learning Rate
def updateAlpha(alpha, decAlpha):
    newAlpha = 0
    newAlpha = round(alpha * decAlpha, 20)
    return newAlpha
# <--- Akhir Bagian Metode LVQ --->


# <--- Bagian Training PSO --->
def train_pso(data, label, row_data, maxIteration, row_particle, w, c1, c2):
    particle = initPartikel(row_particle)
    velocity = initV(row_particle)
    cost = costData(row_data, row_particle, data, particle, label)
    fitness = calculateFitness(cost, row_data)
    pbest = initialPBest(particle)
    gbest, gbest_fitness = setGBest(pbest, fitness)
    iteration = 0
    while iteration < maxIteration:
        velocity = updateVelocity(w, c1, c2, particle, pbest, gbest, velocity)
        particle = updatePosition(particle, velocity)
        cost = costData(row_data, row_particle, data, particle, label)
        old_fitness = fitness
        fitness = calculateFitness(cost, row_data)
        pbest = updatePBest(old_fitness, fitness, pbest, particle)
        gbest, gbest_fitness = setGBest(pbest, fitness)
        iteration += 1
    iterations = iteration
    slices = [10, 10]
    weight_pso = convert(gbest, slices)
    return weight_pso, iterations
# <--- Akhir Bagian Training PSO --->


# <--- Bagian Training LVQ --->
def train_lvq(data, row_data, column_data, weight, row_weight, label, maxEpoch, alpha, minAlpha, decAlpha):
    epoch = 0
    while (epoch < maxEpoch) and (alpha > minAlpha):
        weight = euclideanLVQ(row_data, row_weight, column_data, data, weight, label, alpha)
        alpha = updateAlpha(alpha, decAlpha)
        epoch += 1
    iteration = epoch
    return weight, iteration
# <--- Akhir Bagian Training LVQ --->


# <--- Bagian Training PSO-LVQ --->
def train_psoLvq(data, label, row_data, column_data, maxIterasi, partikel, w, c1, c2, row_weight, epoch_lvq, lr, minimum_lr, pengurang_lr):
    weight_pso, iteration_pso = train_pso(data, label, row_data, maxIterasi, partikel, w, c1, c2)
    weight_psoLvq, epoch_psoLvq = train_lvq(data, row_data, column_data, weight_pso,  row_weight, label, epoch_lvq, lr, minimum_lr, pengurang_lr)
    return weight_psoLvq, iteration_pso
# <--- Akhir Bagian Training PSO-LVQ --->


# <--- Bagian Metode Testing Akurasi --->
def testing(row_data, row_weight, column_data, data, weight, label):
    classification = []
    for i in range(row_data):  # baris data
        sqrt_value = []
        for j in range(row_weight):  # baris weight
            calculate = 0
            for k in range(column_data):  # column weight dan data
                calculate += pow(data[i][k] - weight[j][k], 2)
            sqrt_value.append(math.sqrt(calculate))
        if sqrt_value[0] <= sqrt_value[1]:
            mins = 0
        else:
            mins = 1
        classification.append(mins)

    count, not_disease, disease = 0, 0, 0
    for i in range(len(classification)):
        if classification[i] == label[i]:
            count += 1
        else:
            count += 0
        if classification[i] == 0:
            not_disease += 1
        else:
            disease += 1

    acc = round((count / row_data) * 100, 3)
    return acc, not_disease, disease, classification, count
# <--- Akhir Bagian Metode Testing Akurasi --->


# <--- Bagian Training Bobot PSO-LVQ dan LVQ --->
def training(maxIterasi, partikel, w, c1, c2, epoch_lvq, lr, pengurang_lr, minimum_lr):
    (data_train, data_test, label_train, label_test, row_data_train, row_data_test, column_data) = read_data(r"./static/data/dataset.csv")
    row_weight = 2
    
    # training pso lvq
    weight_psoLvq, iteration_pso = train_psoLvq(data_train, label_train, row_data_train, column_data, maxIterasi, partikel, w, c1, c2, row_weight, epoch_lvq, lr, minimum_lr,pengurang_lr)
    
    # inisialisasi bobot lvq
    weight, weight_label, data_train, label_train = defineWeight(data_train, label_train)
    row_data_train = len(data_train)
    
    # train lvq murni
    weight_lvq, iteration_lvq = train_lvq(data_train, row_data_train, column_data, weight, row_weight, label_train, epoch_lvq, lr, minimum_lr, pengurang_lr)
    return (weight_psoLvq, weight_lvq, iteration_pso, iteration_lvq, data_test, label_test, row_data_test, column_data)
# <--- Akhir Bagian Training Bobot PSO-LVQ dan LVQ --->


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
            (weight_psoLvq, weight_lvq, iteration_pso, iteration_lvq, data, label, row_data, column_data) = training(maxIterasi, partikel, w, c1, c2, epoch_lvq, lr, pengurang_lr, minimum_lr)
            iter_pso.append(iteration_pso)
            iter_lvq.append(iteration_lvq)

            # bagian testing untuk akurasi
            # psoLvq
            (accuracy_psoLvq, not_disease_psoLvq, disease_psoLvq, classification_psoLvq, count_psoLvq) = testing(row_data, 2, column_data, data, weight_psoLvq, label)
            precisionScore_psoLvq = round(precision_score(label, classification_psoLvq, zero_division=0)* 100, 3)
            recallScore_psoLvq = round(recall_score(label, classification_psoLvq, zero_division=0)* 100, 3)
            f1Score_psoLvq = round(f1_score(label, classification_psoLvq, zero_division=0)* 100, 3)

            acc_psoLvq.append(accuracy_psoLvq)
            precision_psoLvq = f"{precisionScore_psoLvq}"
            recall_psoLvq = f"{recallScore_psoLvq}"
            f1_psoLvq = f"{f1Score_psoLvq}"
            not_disease_psoLvq = f"{not_disease_psoLvq}"
            disease_psoLvq = f"{disease_psoLvq}"
            
            # lvq standar
            (accuracy_lvq, not_disease_lvq, disease_lvq, classification_lvq, count_lvq) = testing(row_data, 2, column_data, data, weight_lvq, label)
            precisionScore_lvq = round(precision_score(label, classification_lvq, zero_division=0)* 100, 3)
            recallScore_lvq = round(recall_score(label, classification_lvq, zero_division=0)* 100, 3)
            f1Score_lvq = round(f1_score(label, classification_lvq, zero_division=0)* 100, 3)

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
        print("penyakit stroke lvq = ", disease_lvq)
        print("iter pso : ", iter_pso)
        print("iter lvq : ", iter_lvq)
        
        context = {
            "maxIterasi": maxIterasi,
            "partikel": partikel,
            "koefisien_w": w,
            "koefisien_c1": c1,
            "koefisien_c2": c2,
            "epoch_lvq": epoch_lvq,
            "learning_rate": lr,
            "pengurang_lr": pengurang_lr,
            "minimum_lr": minimum_lr,
            "title": title,
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

# <--- Akhir Bagian Cek Akurasi PSO-LVQ dan LVQ --->