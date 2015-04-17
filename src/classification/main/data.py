class text_label:
    data = []
    kv = {}
    target = []

    def add(self, dt, target):
        self.data.append(dt)
        self.target.append(target)

    def add_all(self, data_input, target_input):
        self.data = data_input
        self.target = target_input
