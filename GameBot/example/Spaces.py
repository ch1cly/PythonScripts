

class actionSpace(krl.Space):
  def __init__(self):
    self.shape = (1,)
  def sample(self, seed=None):
    if seed: random.seed(seed)
    return random.triangular(-1,1)
  def contains(self, x):
    return  abs(x) <= 1

# observation - массив
# допустимые значения можно не описывать.
class observationSpace(krl.Space):
  def __init__(self):
    self.shape = (5,) #
  def sample(self, seed=None): pass
  def contains(self, x): pass


