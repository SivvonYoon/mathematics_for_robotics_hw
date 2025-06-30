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
1. (3-1) https://www.rapidtables.org/ko/math/symbols/Algebra_Symbols.html
2. (3-1) https://leco.tistory.com/4
3. (3-2) https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter20.02-Finite-Difference-Approximating-Derivatives.html
4. (3-2) https://www.youtube.com/watch?v=bo2IPUnXAcg
5. (3-2) https://www.youtube.com/watch?v=vVyiUGqeRhU&t=54s
6. 로봇수학 강의 자료

Problem 3. (5pts)
Given the following one-dimensional (1D) equation
minus d^2u/dx^2, x ∈ [0, 1]
u(0) = 2, u(1) = minus 1.

1. Find an exact solution.

2. Discretize the 1D equation by centered difference scheme with n interior mesh points. 
Form linear system Ax = b.
(Hint: ∆x = xi+1 minus xi = 1 / (n minus 1) and xi = (i minus 1) ∆x)

3. Solve the discrete linear system Ax = b by any numerical linear algebra code 
(e.g., Gaussian elimination by LU factorization) 
for n = 200, 400, 800, 1000, 2000, 4000 and larger  if your computer allows.

• Plot the comparison of the numerical solutions with the exact solution.
• Plot and tabulate the average errors for various n.
• Plot and tabulate CPU computation times for various n.
• Discuss and analyze the results.
(Hint: Average error = sigma (i=1 to n)u_exact(xi) minus unumerical(xi)/n. 
The computational cost is associated with the size of the system.)
"""
import numpy as np
import sympy as sp
from sympy import solve
from sympy.abc import x,y
import scipy.linalg
import matplotlib.pyplot as plt

# 3-1. Find an exact solution
formula_origin = x - 1/2
c1 = sp.Symbol('c1')
integrate_once = sp.integrate(formula_origin, x) + c1
print(integrate_once)
c2 = sp.Symbol('c2')
integrate_twice = sp.integrate(integrate_once, x) + c2
print(integrate_twice)
y = integrate_twice         # from origin formula

equation_from_initial_value_1 = y.subs(x, 0) - 2            # u(0) = 2
equation_from_initial_value_2 = y.subs(x, 1) + 1            # u(1) = -1

solution = solve([equation_from_initial_value_1, equation_from_initial_value_2], (c1, c2)) #연립방정식으로 c1, c2 풀기
y = y.subs(solution) # c1, c2를 y에 대입

print("This is the exact solution of y --> ", y)

# 3-2. Descretize the 1D equation by centered(central?) difference scheme with n interior mesh points.
# Form linear sys Ax = b.

# 먼저 x value Descretize 
# (힌트1) x 변화량인 x_(i+1) - x_i 를 1/(n-1)로 줌. 
# ==> 왜냐면 문제에서 구간을 경계점 포함 n interior mesh points 으로 descretize 했다는 것이다.
# ==> 즉, x0, x1, x2, ..., xn-1 가 x values
def solve_2nd_derivative_with_central_difference(n):    # 3-3 번에서 n을 바꿔가면서 풀려고 n이 input인 함수로 만듦.
    dx = 1/(n-1)                      # 힌트1
    x = np.linspace(0, 1, n-2)    # x 범위 (0, 1)
    
    # 여기서 central difference scheme 을 활용
    ''' 
    ==> u''(x) = ddu / dxx 
               = {u(x_(i+1)) -2*u(x_i) + u(x_(i-1))} / {(∆x)^2}
               = -x + 1/2
    ==> Ax = b 형태로 만들기 위해서 다음과 같이 일단 정리한다
    i-1번째 ddu = u(x_(i+1)) -2*u(x_i) + u(x_(i-1)) = {(∆x)^2}*(-x + 1/2)
    근데 여기서
    A 를 1 -2 1 이 들어가도록 만들어 줄 때, matrix로 만들어 줄 것이라서 
    처음과 끝 항은 (즉 경계항은) 각각 u0랑 un 이 빠지게 된다 => 이건 보고서에서 더 자세히..
    '''
    # 그래서 b의 기본형은 아래와 같음
    b = dx**2 * (-x + 1/2)            # Ax = b 에서의 b vector
    
    # 첫항과 마지막 항은 따로 경계값 더해줌 (u0, un)
    b[0] += 2                         # b vector의 첫번째 원소를 위한 Ax 식..
    b[-1] += -1                       # b vector의 마지막 원소를 위한 Ax 식..
    
    # Ax = b 형태로 만들어주기 위한 A
    A = -np.eye(n - 2, k=-1) +2 * np.eye(n - 2) - np.eye(n - 2, k=1)
    # print(A)
    
    u_points = scipy.linalg.solve(A, b)
    
    x_all = np.linspace(0, 1, n)
    u_all = np.concatenate(([2], u_points, [-1]))
    
    return x_all, u_all

k = 500
x_all, u_all = solve_2nd_derivative_with_central_difference(k)

# 3-1 번 x, y
y_function = sp.lambdify(x,y,'numpy')
x_exact = np.linspace(0, 1, 100)
y_exact = y_function(x_exact)


plt.plot(x_exact, y_exact, 'r-', label="exact solution", linewidth=1.5)    
plt.plot(x_all, u_all, marker='o', label="numerical solution", linewidth=1)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.title(f"Solution with Centered Difference Scheme--> n = {k}")
plt.legend()
plt.show()