import math


class cost_function:  # test cost for only recall now

    def __init__(self):
        self.min_max = []
        self.values = []  # 100 f,a,p,r
        self.cost = []
        self.cost_index = 0

    def add_min_max(self, min_max):
        self.min_max = min_max
        del self.values[:]
        self.cost.append([0, 0, 0])  # cost for f
        self.cost.append([0, 0, 0])  # cost for a
        self.cost.append([0, 0, 0])  # cost for p
        self.cost.append([0, 0, 0])  # cost for r

    def add_values(self, vals):  # [f,a,p,r]
        self.values.append(vals)

    def next(self):
        del self.values[:]
        self.cost_index += 4
        self.cost.append([0, 0, 0])  # cost for f
        self.cost.append([0, 0, 0])  # cost for a
        self.cost.append([0, 0, 0])  # cost for p
        self.cost.append([0, 0, 0])  # cost for r

    def calculate(self):
        [self.calculate_cost(value) for value in self.values]

    def get_cost(self,trail_num,type=1):
        return self.cost[4*trail_num - type][0]  # returns exponential and recall

    def calculate_cost(self, value):
        # print "ZIP",     zip(value, self.min_max, self.cost)
        current_cost = self.cost[self.cost_index:self.cost_index+4]
        for v, m, c in zip(value, self.min_max, current_cost):
            if float(v) > float(m[0]):
                c[0] += 0
                c[1] += 0
                c[2] += 0
            elif float(v) < float(m[1]):
                c[0] += 1
                c[1] += 1
                c[2] += 1
            else:
                #print c[0]
                c[0] += math.exp(float(v) * -1)
                c[1] += 0
                c[2] += (1 / (float(m[1]) - float(m[0]))
                         * (float(v) - float(m[1])) + 1)
