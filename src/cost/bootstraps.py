import numpy

class bootstrap:

	def __init__(self):
		self.alpha_values = {}

	def add(self,values,alpha):
		if alpha in self.alpha_values and len(self.alpha_values[alpha]) >= 100:
			pass

		if alpha in self.alpha_values:
			self.alpha_values[alpha].append(values)
		else:
			self.alpha_values[alpha] = [values]
	def get(self,alpha):
		return self.alpha_values[alpha]






