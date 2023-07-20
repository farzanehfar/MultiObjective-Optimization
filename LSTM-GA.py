# This code was written by the following contributors for the publication, "Data-Driven Multi-Objective Optimisation for Electric Vehicle Charging Infrastructure":

# Farzaneh Farhadi (F.Farhadi2@newcastle.ac.uk): Farzaneh holds an MSc in Computer Science and is currently a PhD student at Newcastle University, UK.

# Shixiao Wang (forainest789@gmail.com): Shixiao has an MSc in Computer Science from Newcastle University, UK.


import os
import random
import time
import pandas as pd
import numpy as np
import math as ma

from tqdm import *
from scipy import spatial
from operator import itemgetter, attrgetter
from collections import Counter
from copy import deepcopy


'''
Initialize LSTM-GA 
'''
class LSTM_GA():
    def __init__(self, type= 6,
                 url = str(),
                 ev_quantity = int(),
                 generation_num = int(),
                 constant_ev_quantity = True,
                 output_dir = str(),
                 worst_case_eval = False
                 ):
        '''
        :param type: charger type list
        :param url:  dataset path
        :param ev_quantity:  Initializing the number of electric cars in the scene
        :param generation_num:  Total number of iterations
        :param constant_ev_quantity: Whether to use a constant number of electric cars throughout
        :param output_dir: Output path
        '''
        self.type  = type
        self.type_list = [[7, 2], [11, 2], [22, 3], [50, 4], [100, 5], [150, 5], [200, 6]]    #available charger types [Charging station capacity, Cost share]
        self.baseMoney = 550  # Basement of  Cost share
        self.dataset = self.load_dataset(url)
        self.boundary = [[54.9641, -1.76835] , [55.059, -1.53996]]  # Using the coordinates of the top-left and bottom-right points to define basic rectangular regions
        self.ev_quantity = ev_quantity
        self.generation_num = generation_num
        self.population = []  #  Population pool of each generation
        self.hunt_count = []  #  Prey captured in each generation: Fully charged electric cars
        self.satisfy_demand = []  # Satisfaction rate in each generation
        self.elits = []*6
        self.worst_case_eval = worst_case_eval
        self.not_satisfied = []  #  Uncaught prey in each generation: Uncharged electric cars
        self.ev_food_list = []   #  Food list in each generation: List of electric cars
        if self.type == 6:
            self.type_use = self.type_list[:-1]
        else:
            self.type_use = self.type_list[:]
        self.sum_charger_num = 0  #  Counting the number of charging stations
        self.avg_work_time = 0    #  Calculating the working time of charging stations
        self.sum_cost = []        #  Total cost in each generation  = Basement of  Cost share x Cost share x  Number
        self.type_count = []      #  Total type of chargers for each generation
        self.constant_ev_quantity = constant_ev_quantity
        self.utilization_rate = []

    '''
    Load data from the training datasets which are the EV power demand from 2035 to 2050 
    '''
    def load_dataset(self,url,sheet_name=0):
        if url.find(".xlsx"):
            return pd.read_excel(url, sheet_name=sheet_name)
        elif url.find(".csv"):
            return pd.read_csv(url)

    '''
    selected suitable data items from dataset for train.
    '''
    def select_items(self):
        try:
            self.latitude = self.dataset['LSOA/DZ centre point latitude']
            self.longitude = self.dataset['LSOA/DZ centre point longitude']
            self.EV_power_demand = self.dataset['Total EV power demand']
            self.vehicles_exit_prob = self.dataset['vehicles percentage']
        except Exception:
            raise("The data columns are not suitable, please check columns")

    '''
    cos-similarity is used to prevent two same chargers are on one position
    '''
    def cos_sim(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)

    '''
    Gene sequence crossover : Exchanging segments of parental genes
    '''
    def crossover(self, geneset3, geneset4, base_prob = 0.2):
        '''
        Imitating mitosis
        :param geneset3: Gene sequence of parent1
        :param geneset4: Gene sequence of parent2
        :param base_prob: The probability of occurrence for gene sequence crossover is 0.8.
        :return: Offspring gene sequences
        '''
        geneset1 = deepcopy(geneset3)
        geneset2 = deepcopy(geneset4)
        Crossover_prob = random.random()
        if Crossover_prob > base_prob:
            l = random.randrange(0, 3, 1)
            temp = geneset1[l]
            geneset1[l] = geneset2[l]
            geneset2[l] = temp
        mutation_prob_1 = random.random()
        if mutation_prob_1 > 0.5 :
            geneset1 = self.mutation(geneset1)
        mutation_prob_2 = random.random()
        if mutation_prob_2 > 0.5 :
            geneset2 = self.mutation(geneset2)
        return geneset1, geneset2

    '''
    Gene sequence mutation: segments of the gene sequence undergo changes.
    1 degree of latitude is approximately equal to 111 kilometers in the England.   
    1 degree of longitude is approximately equal to 85.27 kilometers in the England.
    
    We approximate the distance in kilometers for 1 degree of latitude and longitude here, 
    and set a service range of 5 kilometers radius for the charging stations.
    '''
    def mutation(self,geneset):
        prob_lati = random.random()
        if prob_lati > 0.5:
            mut_lati = geneset[0] * (1 + 0.045 * (np.random.normal(loc=0.0, scale=1.0, size=None)))
        else:
            mut_lati = geneset[0]
        prob_long = random.random()
        if prob_long > 0.5:
            mut_long = geneset[1] * (1 + 0.058 * (np.random.normal(loc=0.0, scale=1.0, size=None)))
        else:
            mut_long = geneset[1]
        prob_type = random.random()
        if prob_type > 0.5:
            mut_type = random.sample(self.type_use, 1)
            gene_type = mut_type[0][0]
            gene_cost = mut_type[0][1]
        else:
            gene_type = geneset[2]
            gene_cost = geneset[3]
        return [mut_lati, mut_long, gene_type, gene_cost, 0, 0, 0]

    '''
    generate the initial population                                                                               
    We place a charging station as a predator at the approximate center of each region.  
    '''
    def initial_population(self):
        start_time = time.time()
        get_food_score = 0
        worktime_score = 0
        generation = 0
        pdbr = tqdm(range(0, self.latitude.size),leave = False, desc = "initializing")
        for i in pdbr:
            type_sample = random.sample(self.type_use, 1)
            # gene : latitude ,longitude,chargerType,chargerCost) # score: hunt_food, work_time, survival time,
            self.population.append([self.latitude[i],self.longitude[i],type_sample[0][0],type_sample[0][1], get_food_score, worktime_score, generation])

        print(f'finish initial using time {(time.time()-start_time)}')
        return self.population

    '''
    Generating food (prey) for each generation: Electric Vehicals
    '''
    def generate_food(self,last_ev_food_count):
        # Clearing and regenerating
        self.ev_food_list = []
        for areas in range(0, len(self.vehicles_exit_prob)):
            if random.random() > 0.5 :
                areas_ev_quantity = ma.ceil(self.vehicles_exit_prob[areas] * last_ev_food_count)
            else:
                areas_ev_quantity = ma.floor(self.vehicles_exit_prob[areas] * last_ev_food_count)
            assert areas_ev_quantity != 0
            for ev in range(areas_ev_quantity):
                '''
                1 degree of latitude is approximately equal to 111 kilometers in the England.
                1 degree of longitude is approximately equal to 85.27 kilometers in the England.
                
                We approximate the distance in kilometers for 1 degree of latitude and longitude here, 
                and generate possible demands around the center of the region with a radius of 10 kilometers 
                using a normal distribution.                                                                        
                '''
                ev_lati = self.latitude[areas] * (1 + 0.090 * np.random.normal(loc=0.0, scale=1.0, size=None))
                ev_long = self.longitude[areas] * (1 + 0.116 * np.random.normal(loc=0.0, scale=1.0, size=None))
                '''
                We idealize the worst-case and best-case scenarios to evaluate a reasonable interval. 
                Worst case scenario is considered to be heavy daily charging demand for EVs
                Base case scenario considered some daily charging needs for EVs
                '''
                if self.worst_case_eval:
                    ev_demand = self.EV_power_demand[areas]/areas_ev_quantity * (1 + np.random.normal(loc=0.0, scale=1.0, size=None))
                else:
                    ev_demand = self.EV_power_demand[areas]/areas_ev_quantity * (2 - random.random())

                '''
                Abnormally large or small charging demands will be filtered out and regenerated.
                240kw is the estimated top battery capacity
                The charging demand of [0, 1]kw will be considered as a EV that continues to stay after charging
                '''
                while(ev_demand > 240 or ev_demand <= 0):
                    ev_demand = self.EV_power_demand[areas] * (1 + np.random.normal(loc=0.0, scale=1.0, size=None))

                self.ev_food_list.append([ev_lati, ev_long, ev_demand])
        return self.ev_food_list

    '''
    The predation process involves the charging stations (predators) capturing the EV demands (prey) within their service range. 
    The charging stations determine which demands to capture based on certain criteria, such as proximity, demand size. 
    Once a EV demand is captured by a charging station, it is fulfilled by providing the required charging service.
    '''
    def hunt(self):
        hunt_food_num = 0
        for i in range(0, len(self.ev_food_list)):
            distance = []
            for j in range(0, len(self.population)):
                distance.append([np.power((np.abs(self.population[j][0] - self.ev_food_list[i][0]) + np.abs(self.population[j][1] - self.ev_food_list[i][1])),2),j])
            distance.sort()
            for k in range(0, len(distance)):
                work_hour_score = self.population[distance[k][1]][5] + \
                                  self.ev_food_list[i][2] / (self.population[distance[k][1]][2])
                if((work_hour_score) < 20 * 0.8):
                    self.population[distance[k][1]][5] += self.ev_food_list[i][2] / self.population[distance[k][1]][2]
                    self.population[distance[k][1]][4] += 1
                    hunt_food_num += 1
                    break
        self.hunt_count.append(hunt_food_num)
        self.satisfy_demand.append(hunt_food_num/self.ev_quantity)

    '''
    Filtering out coordinates with exceptional values.
    '''
    def check_list(self, generation_now:int, is_offspring=False):
        data_set = []
        for charger in range(len(self.population)):
            for charger_other in range(len(self.population)):
                if charger_other == charger:
                    break
                if self.cos_sim(self.population[charger], self.population[charger_other]) == 1.:
                    self.population[charger_other][0] = 0
            is_in_boundry_x = self.boundary[0][0] < self.population[charger][0] < self.boundary[1][0]
            is_in_boundry_y = self.boundary[0][1] < self.population[charger][1] < self.boundary[1][1]
            if is_in_boundry_x and is_in_boundry_y:
                if (generation_now == self.generation_num):
                    if self.population[charger][4] > 0 and self.population[charger][5] > 0:
                        data_set.append(self.population[charger])
                elif is_offspring:
                    self.population[charger][4] = 0
                    self.population[charger][5] = 0
                    data_set.append(self.population[charger])
                else:
                    if self.population[charger][4] > 0 and self.population[charger][5] > 0:
                        self.population[charger][4] = 0
                        self.population[charger][5] = 0
                        data_set.append(self.population[charger])
        self.population = data_set

    '''
    Calculate the Utilization rate
    '''
    def get_utilization(self):
        w1 = 0.6
        w2 = 0.4
        Tm = 10
        self.utilization_rate.append(self.satisfy_demand[-1] * w1 + (1-((self.avg_work_time - Tm) / Tm) ** 2) * w2)

    '''
    Generating offspring through genetic algorithms.
    '''
    def get_offsprings(self, generation_now:int):
        offspring = []
        for single in self.population:
            other = random.sample(self.population, 1)
            other = other[0]
            if single[6] < generation_now - 4:  # it means an individual is too old to generate offsprings
                    break

            # The elite individuals can have two offsprings
            elif single[4] >= self.hunt_food_score[2] and single[5] >= self.work_hour_rank[2]:
                        self.elits.append(single)
                        geneset1, geneset2 = self.crossover(single, other)
                        geneset1[6] = generation_now
                        geneset2[6] = generation_now
                        offspring.append(geneset1)
                        offspring.append(geneset2)
            # The normal individuals can have one or two offsprings
            elif single[4] >= self.hunt_food_score[1] or single[5] >= self.work_hour_rank[1]:
                        geneset1, geneset2 = self.crossover(single, other)
                        geneset1[6] = generation_now
                        geneset2[6] = generation_now
                        survivor_prob = random.random()
                        if survivor_prob > 0.5:
                            offspring.append(geneset1)
                        offspring.append(geneset2)

            # The poor individuals can have one offsprings
            elif single[4] >= self.hunt_food_score[0] or single[5] >= self.work_hour_rank[0]:
                        geneset1, geneset2 = self.crossover(single, other)
                        geneset1[6] = generation_now
                        geneset2[6] = generation_now
                        survivor_prob = random.random()
                        if survivor_prob > 0.4:
                            survivor_prob = random.random()
                            if survivor_prob > 0.5:
                                offspring.append(geneset1)
                            else:
                                offspring.append(geneset2)
            # The unfit individuals will be expelled
            else:
                single[0] = 0
        self.population += offspring
        self.check_list(generation_now=generation_now, is_offspring=True)

    '''
    Categorize into ranks
    '''
    def get_rank(self,generation_now):

        #hunt_food_score
        self.population.sort(key=itemgetter(4), reverse=True)
        hunt_food_score_limit_top = self.population[0][4]
        hunt_food_score_limit_1 = hunt_food_score_limit_top * 0.2
        hunt_food_score_limit_2 = hunt_food_score_limit_top * 0.5
        hunt_food_score_limit_3 = hunt_food_score_limit_top * 0.9

        #work_hour_score
        self.population.sort(key=itemgetter(5),reverse=True)
        work_hour_score_limit_top = self.population[0][5]
        work_hour_score_limit_1 = work_hour_score_limit_top * 0.2
        work_hour_score_limit_2 = work_hour_score_limit_top * 0.5
        work_hour_score_limit_3 = work_hour_score_limit_top * 0.9

        if self.elits != []:
            temp = []
            '''
            There is a chance that elites beyond a certain number of iterations will be forgotten by the population.
            '''
            for elit in self.elits:
                if elit[6]>= generation_now - 5:
                     temp.append(elit)
                elif random.random() > 0.5:
                     temp.append(elit)
            self.elits = temp

            elits_mean = np.array(self.elits).mean(axis=0)
            avg_time = elits_mean[5]
            self.avg_work_time = avg_time
            avg_hunt = elits_mean[4]
            '''
            The rank is determined by the most dominant elite, the ruler, 
            and a few select members of the population, collectively making the decision.
            '''
            hunt_food_score_limit_1 = 0.8 * hunt_food_score_limit_1 + 0.2 * avg_hunt
            hunt_food_score_limit_2 = 0.9 * hunt_food_score_limit_2 + 0.1 * avg_hunt
            hunt_food_score_limit_3 = 0.99 * hunt_food_score_limit_3 + 0.01 * avg_hunt

            work_hour_score_limit_1 = 0.8 * work_hour_score_limit_1 + 0.2 * avg_time
            work_hour_score_limit_2 = 0.9 * work_hour_score_limit_2 + 0.1 * avg_time
            work_hour_score_limit_3 = 0.99 * work_hour_score_limit_3 + 0.01 * avg_time
        '''
        However, the rank is influenced by the quantity of food in the environment. 
        For example, when food is scarce and the population size is large, 
        the competition is intense, leading to a stronger hierarchy. 
        On the other hand, when food is abundant and the population size is small, 
        the competition is weaker, resulting in a more relaxed hierarchy.
        '''
        self.work_hour_rank = [self.satisfy_demand[-1] * item for item in [work_hour_score_limit_1, work_hour_score_limit_2, work_hour_score_limit_3]]
        self.hunt_food_score = [self.satisfy_demand[-1] * item for item in[hunt_food_score_limit_1, hunt_food_score_limit_2, hunt_food_score_limit_3]]
        
                    
    def write_data(self):
        try:
            self.sum_type = pd.DataFrame(data=self.train_total_population,
                                         columns=['Chargers Number'])
            self.sum_cost = pd.DataFrame(data=self.sum_cost,
                                         columns=['Toral Cost'])
            self.ev_quantity = pd.DataFrame(data=self.train_total_evfood,
                                         columns=['Total EV Quantity'])
            self.hunt_count = pd.DataFrame(data=self.hunt_count,
                                            columns=['Satisfied EV Quantity'])
            self.satisfy_demand = pd.DataFrame(data=self.satisfy_demand,
                                                columns=['Satisfied Percentage'])
            self.utilization_rate = pd.DataFrame(data=self.utilization_rate,
                                                  columns=['Utilization'])
            df_concat = pd.concat([self.prpotion, self.sum_type, self.sum_cost, self.ev_quantity, \
                                   self.hunt_count, self.satisfy_demand, self.utilization_rate ], join='inner', axis=1)
            writer = pd.ExcelWriter(output_dir)
            self.population_details.to_excel(
                                             writer,
                                             sheet_name="population_details",
                                             index=False,
                                            )
            df_concat.to_excel(
                               writer,
                               sheet_name="train_details",
                               index=False,
                                )
            writer.save()

        except:
            raise ("Data Output is wrong, please check it again")

    def count_type(self):
        self.population_details = pd.DataFrame(data=self.population,
                                  columns=['Latitude', 'Longitude', 'Type of charging pile (kW)',
                                           'Economic costs', 'EV Number of Charging Posts Serviced',
                                           'Charging Posts Operating Hours', 'generation'])
        self.type_count.append(list((Counter(self.population_details['Type of charging pile (kW)'])).values()))
        if len(self.type_use) == 6:
            self.prpotion = pd.DataFrame(data=self.type_count,
                                columns=['7kw', '11kw', '22kw', '50kw', '100kw', '150kw'])
        else:
            self.prpotion = pd.DataFrame(data=self.type_count,
                                         columns=['7kw', '11kw', '22kw', '50kw', '100kw', '150kw', '200kw'])
        self.sum_cost.append(sum(self.population_details['Economic costs']))

    def train(self):
        self.train_total_population = []
        self.train_total_evfood = []
        self.select_items()
        self.population = self.initial_population()
        pdbr = tqdm(range(0, self.generation_num + 1), leave = False, desc = "training")
        strat_time = time.time()
        self.generate_food(self.ev_quantity)
        for generation_now in pdbr:
            self.hunt()
            is_last_generation = generation_now == self.generation_num
            if is_last_generation:
                self.check_list(generation_now)
                self.train_total_population.append(len(self.population))
                self.train_total_evfood.append(len(self.ev_food_list))
                self.count_type()
                self.get_utilization()
                self.write_data()
            else:
                self.get_rank(generation_now)
                self.get_offsprings(generation_now)
                self.train_total_population.append(len(self.population))
                self.train_total_evfood.append(len(self.ev_food_list))
                self.count_type()
                self.get_utilization()
                if self.constant_ev_quantity:
                    self.generate_food(self.ev_quantity)
                else:
                    self.generate_food(self.train_total_evfood[-1])


input_dir = r"F:\LSTM-GA\LSTM-GA\Train_dataset"

year_num = {
    '2042': 134606,
    '2046': 145345,
    '2048': 146617,
    '2050': 160403,

}

if os.path.isdir(input_dir):
    files = os.listdir(input_dir)

    for file in files:
        url = os.path.join(input_dir, file)
        file_name = file.split(".xlsx")[0]
        output_dir = f'./result_{file_name}.xlsx'
        year = file_name.split("_")[0]
        ev_quantity = year_num[file_name.split("_")[1]]
        print(f'start training {year}, total EV quantity is {ev_quantity}')
        lstm_ga = LSTM_GA(
            url=url,
            ev_quantity=ev_quantity,
            generation_num=100,
            output_dir = output_dir,
        )
        lstm_ga.train()
    print(output_dir," finished train")
else:
    print("Please check input dir")




