from random import random
import math
from itertools import islice

class PSO:
    # <--- Bagian Metode PSO --->
    # Inisialisasi kecepatan awal = 0
    def initV(self, size):
        column = 20
        velocity = [[0] * column] * size
        for i in range(size):
            for j in range(column):
                velocity[i][j] = 0
        return velocity

    # Inisialisasi partikel awal secara random
    def initPartikel(self, size):
        column = 20
        position = []
        for i in range(size):
            row = []
            for j in range(column):
                row.append(round(random(), 3))
            position.append(row)
        return position

    # Optimasi mencari nilai cost data
    def hitungCost(self, row_data, row_particle, data, particle, label):
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
        # cek perbedaan hasil jarak minimal(cost) dengan target data(kelas sebenarnya)
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

    # Menghitung nilai fitness
    def calculateFitness(self, cost_value, row_data):
        fitness_value = []
        for i in range(len(cost_value)):
            fitness = 0
            fitness = (row_data - cost_value[i]) / row_data
            fitness_value.append(fitness)
        return fitness_value

    # Inisialisasi Pbest
    def initialPBest(self, particle):
        pBest = particle
        return pBest

    # Update Pbest
    def updatePBest(self, oldFitness, newFitness, oldPBest, newPosition):
        new_pbest = []
        for i in range(len(oldFitness)):
            if oldFitness[i] > newFitness[i]:
                new_pbest.append(oldPBest[i])
            elif oldFitness[i] == newFitness[i]:
                new_pbest.append(oldPBest[i])
            else:
                new_pbest.append(newPosition[i])
        return new_pbest

    # Set Gbest
    def setGBest(self, pbest, fitness):
        index = fitness.index(max(fitness))
        return pbest[index], fitness[index]

    # Uupdate kecepatan partikel
    def updateVelocity(self, w, c1, c2, particle, pbest, gbest, velocity):
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

    # Update posisi partikel
    def updatePosition(self, particle, newVelocity):
        newParticle = []
        for i in range(len(particle)):
            temp = []
            for j in range(len(particle[i])):
                temp.append(round(particle[i][j] + newVelocity[i][j], 3))
            newParticle.append(temp)
        return newParticle

    # Convert gbest menjadi 2 list untuk masing" kelas
    def convert(self, lst, var_lst):
        it = iter(lst)
        return [list(islice(it, i)) for i in var_lst]

    # <--- Bagian Training PSO --->
    def train_pso(self, data, label, row_data, maxIteration, row_particle, w, c1, c2):
        particle = self.initPartikel(self, row_particle)
        velocity = self.initV(self, row_particle)
        cost = self.hitungCost(self, row_data, row_particle, data, particle, label)
        fitness = self.calculateFitness(self, cost, row_data)
        pbest = self.initialPBest(self, particle)
        gbest, gbest_fitness = self.setGBest(self, pbest, fitness)
        iteration = 0
        while iteration < maxIteration:
            velocity = self.updateVelocity(self, w, c1, c2, particle, pbest, gbest, velocity)
            particle = self.updatePosition(self, particle, velocity)
            cost = self.hitungCost(self, row_data, row_particle, data, particle, label)
            old_fitness = fitness
            fitness = self.calculateFitness(self, cost, row_data)
            pbest = self.updatePBest(self, old_fitness, fitness, pbest, particle)
            gbest, gbest_fitness = self.setGBest(self, pbest, fitness)
            iteration += 1
        iterations = iteration
        slices = [10, 10]
        weight_pso = self.convert(self, gbest, slices)
        return weight_pso, iterations
