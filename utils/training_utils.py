class EMAModel:
    def __init__(self, parameters, power=0.75):
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.power = power
        self.collected_params = None

    def step(self, parameters):
        for s, p in zip(self.shadow_params, parameters):
            s.data.sub_((1.0 - self.power) * (s.data - p.data))

    def copy_to(self, parameters):
        for p, s in zip(parameters, self.shadow_params):
            p.data.copy_(s.data)
            