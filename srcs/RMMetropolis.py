import numpy as np
from .utils import dual_likelihood, single_likelihood
import seaborn as sns
import matplotlib.pyplot as plt

class RMMetropolis:
    def __init__(self, totalCategory, totalWords, minIndex, maxIndex, burn_in, num_sample):
        self.totalWords = totalWords
        self.totalCategory = totalCategory
        self.cat_mean = []
        self.ind_cat = []
        self.log_like = []
        self.minIndex = minIndex
        self.maxIndex = maxIndex
        self.burn_in = burn_in
        self.num_sample = num_sample

    def get_cat_mean(self):
        return self.cat_mean

    def get_ind_cat(self):
        return self.ind_cat
    
    def get_log_like(self):
        return self.log_like

    def initialize(self):
        #randomly initialize category means and individual category assignments
        self.cat_mean = [[(i * 100) / self.totalCategory 
                          for i in range(self.totalCategory)]]
        
        self.ind_cat.append((np.random.choice(
                    np.arange(0,self.totalCategory,1),
                    size = self.totalWords, 
                    p = np.repeat(1/self.totalCategory,self.totalCategory))))
        
        #locate the words with the min mean score and max mean score
        self.ind_cat[0][self.minIndex] = 0
        self.ind_cat[0][self.maxIndex] = self.totalCategory-1

        self.log_like = [[] for i in range(self.totalWords)]

    def sample(self,data,varianceHyper):
        for i in range(1,self.num_sample):
            #proposal sampling
            if (i % 1000 == 0):
                print(f"Current Sample: {i}")
            temp_cat_mean = sorted(abs(np.random.normal(
                loc = self.cat_mean[i-1],
                scale = varianceHyper
                )))
            temp_ind_cat=((np.random.choice(
                np.arange(0,self.totalCategory,1),
                size = self.totalWords, 
                p = np.repeat(1/self.totalCategory,self.totalCategory
                ))))
            temp_ind_cat[self.minIndex] = 0
            temp_ind_cat[self.maxIndex] = self.totalCategory-1
            #piecewise acceptance/rejection using data likelihood directly:
            accepted = []
            rejected = []
            total = []
            for s in sorted(np.arange(0,40,1)):
                dataList = data[s]
                if (np.std(dataList) == 0):
                    data_var = 1.0
                else:
                    data_var = np.std(dataList)
                cur_like, prev_like = dual_likelihood(
                                                        temp_ind_cat[s],
                                                        dataList,
                                                        self.cat_mean[i-1][self.ind_cat[i-1][s]],
                                                        temp_cat_mean[temp_ind_cat[s]],
                                                        data_var,
                                                        self.totalCategory
                                                        )
                likelihood_ratio = cur_like - prev_like
                acceptance_ratio = np.log(np.random.uniform(0,1,size = 1))[0]
                self.log_like[s].append(cur_like)
                #accept or reject
                if (acceptance_ratio < likelihood_ratio):
                    accepted.append(temp_ind_cat[s])
                    total.append(temp_ind_cat[s])
                else:
                    rejected.append(self.ind_cat[i-1][s])
                    total.append(self.ind_cat[i-1][s])
            #do category mean parameter update
            for categories in range(self.totalCategory):
                if categories not in accepted:
                    temp_cat_mean[categories] = self.cat_mean[i-1][categories]
            self.cat_mean.append(temp_cat_mean)
            self.ind_cat.append(total)    

    def plotting(self):
        for i in range(self.totalCategory):
            sns.kdeplot(np.array(self.cat_mean)[self.burn_in:,i])
        plt.xlabel("Category Mean")
        plt.ylabel("Count")
        plt.title("Category Mean Distribution for " + str(self.totalCategory) + " Categories")
        plt.show()

    def DIC(self,data):
        #Assign each words to its category
        category_list = []
        data_var = 0
        best_estimator = np.mean(np.array(self.cat_mean)[self.burn_in:,],axis = 0)
        for i in range(self.totalWords):
            cur_category = np.bincount(np.array(self.ind_cat)[self.burn_in:,i])
            category_list.append(np.argmax(cur_category))
        firstPart = 0
        for s in range(self.totalWords):
            if (np.std(data[s]) == 0):
                data_var = 1
            else:
                data_var = np.std(data[s])
            firstPart += single_likelihood(
                category_list[s],
                data[s], 
                best_estimator[category_list[s]], 
                data_var,
                self.totalCategory)
        tempSecond = 0
        for s in range(self.num_sample - self.burn_in):
            for i in range(self.totalWords):
                if (np.std(data[i]) == 0):
                    data_var = 1
                else:
                    data_var = np.std(data[i])
                tempSecond += single_likelihood(
                    category_list[i],
                    data[i], 
                    self.cat_mean[self.burn_in + s][category_list[i]], 
                    data_var,
                    self.totalCategory)
        tempSecond = tempSecond/(self.num_sample - self.burn_in)
        pdic = 2*firstPart -2* tempSecond
        thirdPart = pdic + (-2*tempSecond)
        return thirdPart

    def PDIC(self,data):
        #assign each word into its category
        category_list = []
        data_var = 0
        best_estimator = np.mean(np.array(self.cat_mean)[self.burn_in:,],axis = 0)
        for i in range(self.totalWords):
            cur_category = np.bincount(self.ind_cat[15000:,i])
            category_list.append(np.argmax(cur_category))
        #calculate the first part of the formula
        firstPart = 0
        for s in range(self.totalWords):
            if (np.std(data[s]) == 0):
                data_var = 1
            else:
                data_var = np.std(data[s])
            firstPart += single_likelihood(
                category_list[s],
                data[s], 
                best_estimator[category_list[s]], 
                data_var,
                self.totalCategory)
            
        #calculate the second part of the formula
        tempSecond = 0
        for s in range(self.num_sample - self.burn_in):
            for i in range(self.totalWords):
                if (np.std(data[i]) == 0):
                    data_var = 1
                else:
                    data_var = np.std(data[i])
                tempSecond += single_likelihood(
                    category_list[i],
                    data[i], 
                    self.cat_mean[self.burn_in + s][category_list[i]], 
                    data_var,
                    self.totalCategory)
        tempSecond = tempSecond/(self.num_sample - self.burn_in)
        pdic = 2*firstPart -2* tempSecond
        return pdic





