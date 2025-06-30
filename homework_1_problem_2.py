# HW1 of ROSS705 Math. for Robotics

"""
경북대학교 전자전기공학부 박사 과정 1학기 윤시원
student id 2025000320

python 3.7.0
numpy 1.21.6

주석을 포함한 코드 전체를 직접 작성하였습니다.

참고 문서, 자료
1. https://zzziito.tistory.com/m/78
2. https://jimmy-ai.tistory.com/106
3. 로봇수학 강의 자료 

Problem 2. (2pts)
Let us consider the problem of finding an equation of the line that best approximates the following four points
: (1, 3.5),(2, 4.3),(3, 7.2), and (4, 8).
(Hint: Assume the equation of the line is in the form ax + by + c = 0. 
We can express this equation in matrix form AX = 0, where X = [a, b, c]T. 
We want to find the values of a, b, and c that satisfy the homogeneous system AX = 0.)
"""

import numpy as np
import matplotlib.pyplot as plt

# 주어진 4개의 points (x_i, y_i)
p1 = np.array([1, 3.5])
p2 = np.array([2, 4.3])
p3 = np.array([3, 7.2])
p4 = np.array([4, 8.0])

# 네 점을 합쳐서 matrix A 만들기 (AX = 0)
A = np.array([p1, p2, p3, p4])
# print(A)

# X = [a, b, c].T -> 3 x 1 vector 이다. (AX = 0 로 풀려면 A가 n x 3 matrix여야 함.)
A = np.hstack([A, np.ones((A.shape[0], 1))])
# print(A)

# 주어진 points 의 개수가 미지수 a, b, c의 개수보다 많으므로 -> overdetermined system
# AX=0, overdetermined system -> SVD 를 이용 (특이값분해 후 V의 마지막 column을 이용하기 위해서 => 이유는 보고서에 자세히 설명.)
U, sigma, V_T = np.linalg.svd(A)        # V_T == V.T
# print(V_T)

# V의 마지막 column을 이용해야 한다 -> V_T(V.T)의 마지막 row
# print(V_T[-1])
X = V_T[-1]
a = X[0]
b = X[1]
c = X[2]
print(a, b, c)

# plot 해서 확인
x_axis = np.linspace(0, 5, 1000)        # 주어진 네 포인트 (x_i, y_i)의 x_i 값이 1, 2, 3, 4 여서 범위 0~5
y_axis = (-a * x_axis - c) / b          # ax + by + c = 0 -> y = (-ax - c) / b

plt.plot(x_axis, y_axis, 'r-', label='line with best approximates')
plt.scatter(A[:, 0], A[:, 1], color='blue', label='Points')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('This is the best approximation.')
plt.show()