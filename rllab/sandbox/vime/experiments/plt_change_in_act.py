import numpy as np
import matplotlib.pyplot as plt
import re

f = open("data/logs_trpo_sim_physics_st_sz_0.010_sd_0_db_550_max_900/debug.log","r")
contents =f.readlines()
for line in contents:
  print(line)
