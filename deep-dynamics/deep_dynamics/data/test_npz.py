import numpy as np
 
# NPY 파일에서 데이터 로드
loaded_array = np.load('/app/deep_dynamics/data/DYN-PP-ETHZ_5.npz')
 
array1 = loaded_array['features']
array2 = loaded_array['labels']
 
# print(array1)
print(array2)