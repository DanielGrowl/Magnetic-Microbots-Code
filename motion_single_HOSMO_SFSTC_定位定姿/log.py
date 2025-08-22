import draw
import pickle
import ppc
import numpy as np

# 从文件中读取
with open('model_circle.pkl', 'rb') as file:
    magnetic_model = pickle.load(file)

draw.draw_log(magnetic_model.ref_time, magnetic_model.ref_x, magnetic_model.ref_y, magnetic_model.ref_vx, magnetic_model.ref_vy, magnetic_model)
