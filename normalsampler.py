import torch

class NormalSampler:

    def __init__(self, morphableModel):
        self.morphableModel = morphableModel

    def _sample(self, n, variance, std_multiplier = 1):
        std = torch.sqrt(variance) * std_multiplier
        std = std.expand((n, std.shape[0]))
        q = torch.distributions.Normal(torch.zeros_like(std).to(std.device), std * std_multiplier)
        samples = q.rsample()
        return samples

    def sampleShape(self, n, std_multiplier = 1):
        return self._sample(n, self.morphableModel.shapePcaVar, std_multiplier)

    def sampleExpression(self, n, std_multiplier=1):
        return self._sample(n, self.morphableModel.expressionPcaVar, std_multiplier)

    def sampleAlbedo(self, n, std_multiplier=1):
        return self._sample(n, self.morphableModel.diffuseAlbedoPcaVar, std_multiplier)

    def sample(self, shapeNumber = 1):
        shapeCoeff = self.sampleShape(shapeNumber)
        expCoeff = self.sampleExpression(shapeNumber)
        albedoCoeff = self.sampleAlbedo(shapeNumber)
        return shapeCoeff, expCoeff, albedoCoeff

