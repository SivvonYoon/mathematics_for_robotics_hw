# HW1 of ROSS705 Math. for Robotics

"""
경북대학교 전자전기공학부 박사 과정 1학기 윤시원
student id 2025000320

python 3.7.0
numpy 1.21.6
sympy 1.10.1
mpmath 1.3.0

주석을 포함한 코드 전체를 직접 작성하였습니다.

참고 문서, 자료
1. 
2. 로봇수학 강의 자료

Problem 4. (2pts)
Use the Adams Bashforth Moulton method to approximate y(0.4), where
y(x) is the solution of initial-value problem:
y' = 4x minus 2y, (1)
y(0) = 2.
(Hint: Use h = 0.1 and the RK4 method to compute y1, y2, and y3.)
"""
import numpy as np

# Adams bashforth moulton method는 크게 2 part 로 나뉨
# (part1) ==> RK4 (Runge-Kutta 4 points method)

h = 0.1         # step size 전제조건으로 줌 (문제에서)

def function_y_prime(x,y):
    return 4*x - 2*y
    
def this_is_function_k1(x,y):
    return function_y_prime(x,y)

def this_is_function_k2(x,y, k1):
    return function_y_prime(x + h/2, y + (h/2)*k1)

def this_is_function_k3(x,y,k2):
    return function_y_prime(x + h/2, y + (h/2)*k2)

def this_is_function_k4(x,y,k3):
    return function_y_prime(x + h, y + h*k3)

x = 0
y = 2
x_list = [x]
y_list = [y]

for i in range(3):         
    # x=0.4일때 구하면 되는 거니까... step size 0.1이니깐 초기값 포함 x가 0, 0.1, 0.2, 0.3 일때 y값 알면됨
    
    # 룬지쿠타법 썼음 
    k1 = this_is_function_k1(x,y)
    k2 = this_is_function_k2(x,y,k1)
    k3 = this_is_function_k3(x,y,k2)
    k4 = this_is_function_k4(x,y,k3)
    
    # x,y값 갱신함
    x += h
    y += (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    x_list.append(x)
    y_list.append(y)

x_array = np.array(x_list, dtype=float)
y_array = np.array(y_list, dtype=float)
x0, x1, x2, x3 = x_array
y0, y1, y2, y3 = y_array
print(x_array)
print(y_array)

# 이제 아담스 배쉬포스 몰튼 방법 파트 2
# (part2-1) predictor
y4_predictor = y3 + (h/24)*(55*function_y_prime(x3, y3) - 59*function_y_prime(x2, y2) 
                            + 37*function_y_prime(x1, y1) -9*function_y_prime(x0, y0))
print("this is predictor of adams-bashforth-moulton --> ",y4_predictor)
# (part2-2) corrector
y4_corrector = y3 + (h/24)*(9*function_y_prime(0.4, y4_predictor) + 19*function_y_prime(x3, y3) 
                            -5*function_y_prime(x2, y2) +function_y_prime(x1, y1))
print("this is the answer--> corrector of y(0.4) : ", y4_corrector)