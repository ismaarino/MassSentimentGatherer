import json

objecte = lambda: None
objecte2 = lambda: None
setattr(objecte, 'eee', 'kkk')
print(objecte.eee)

setattr(objecte2, "eee", getattr(objecte, "eee"))
print(objecte2.eee)
