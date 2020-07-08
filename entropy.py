#расчет информационной энтропии

import pandas as pd
import numpy as np
import math as m

data = pd.read_csv('cats.csv')

E_sh_sob=(1/1)*m.log2((1/1)) - 0
E_sh_kot=-(4/9)*m.log2((4/9)) - (5/9)*m.log2((5/9))
E_gav_sob=0 - (5/5)*m.log2((5/5))
E_gav_kot=-(4/5)*m.log2((4/5)) - (1/5)*m.log2((1/5))
E_laz_sob=0 - (6/6)*m.log2((6/6))
E_laz_kot=-(4/4)*m.log2((4/4)) - 0


E_Y_X = 4/10 * E_laz_sob + 6/10 * E_laz_kot
E_group = 0.97 - E_Y_X
round(E_group, 2)






