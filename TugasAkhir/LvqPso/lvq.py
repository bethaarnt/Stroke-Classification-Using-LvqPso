import numpy as np
import math

class LVQ:
    # <--- Bagian Metode LVQ --->
    # Inisialisasi weight lvq random
    def defineWeight(self, data_train, label_train):
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
            
    # Menghitung jarak euclidean
    def euclideanLVQ(self, row_data, row_weight, column_data, data, weight, label, alpha):
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

    # Update Learning Rate
    def updateAlpha(self, alpha, decAlpha):
        newAlpha = 0
        newAlpha = round(alpha * decAlpha, 20)
        return newAlpha

    # <--- Bagian Training LVQ --->
    def train_lvq(self, data, row_data, column_data, weight, row_weight, label, maxEpoch, alpha, minAlpha, decAlpha):
        epoch = 0
        while (epoch < maxEpoch) and (alpha > minAlpha):
            weight = self.euclideanLVQ(self, row_data, row_weight, column_data, data, weight, label, alpha)
            alpha = self.updateAlpha(self, alpha, decAlpha)
            epoch += 1
        iteration = epoch
        return weight, iteration

    # <--- Bagian Metode Testing Akurasi --->
    def testing(self, row_data, row_weight, column_data, data, weight, label):
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