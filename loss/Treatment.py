from torch import Tensor
import torch


class Treatment:
    def __init__(self, absorptionRate, decayRate, dose, firstDoseTime, dosesNum, timeBetweenDoses):
        self.absorptionRate = absorptionRate
        self.decayRate = decayRate
        self.dose = dose
        self.firstDoseTime = firstDoseTime
        self.dosesNum = dosesNum
        self.timeBetweenDoses = timeBetweenDoses
    
    def __call__(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        f = torch.zeros_like(t)
        for i in range(self.dosesNum):
            peak_center = self.firstDoseTime + i * self.timeBetweenDoses
            absorptionPhase = self.dose * torch.exp(-self.absorptionRate * (t - peak_center) ** 2) * (t<=peak_center)
            decayPhase = self.dose * torch.exp(-self.decayRate * (t - peak_center) ** 2) * (t>peak_center)
            f += absorptionPhase + decayPhase
        return f
