# HW1 of ROSS705 Math. for Robotics

"""
경북대학교 전자전기공학부 박사 과정 1학기 윤시원
student id 2025000320

python 3.7.0
numpy 1.21.6

주석을 포함한 코드 전체를 직접 작성하였습니다.

참고 문서
1. https://scicoding.com/how-to-calculate-matrix-exponential-in-python/
2. 로봇수학 강의 자료

Problem 1. (1pt)
Show that if A is skew-symmetric, then exp(A) is orthogonal.
(Hint: Use the definition of exp(A).)
"""

import numpy as np
from scipy.linalg import expm

# skew-symmetric matrix 생성 (랜덤)
# 랜덤 k x k mtx 먼저 생성한다
k = 3                           # 3 x 3 matrix가 되도록 임의 지정
a = np.random.randn(k, k).astype(np.float64)

# skew-symmetric mtx. A 생성
A = a - a.T

# A 가 skew-symmetric이 맞는지 검사 (정의)
if np.allclose(A.T, -A, atol=1e-10): # A 각 요소를 정수 포함한 실수로 랜덤생성해서 혹시 모를 오차 때문에 allclose로 검사함
    print(A)
    print("A is skew-symmetric")
    
    # A가 skew-symmetric matrix 일 때, exponential(A)가 orthogonal 한지 검사 (문제에서 구하고자 하는 것)
    # 힌트 : exp(A)의 정의 이용하기 -> taylor's series를 이용한 정의를 사용하는 것으로 이해함.
    
    # exp(A)의 초기 설정: matrix A와 size가 동일한 identity matrix (테일러 시리즈 첫 시작이 A^0=I)
    exp_A = np.eye(A.shape[0], dtype=np.float64)
    
    # A를 계속 곱해나가기 위한 초기 설정
    A_power = np.eye(A.shape[0], dtype=np.float64)
    
    # 테일러 시리즈 각 항의 분모로 들어갈 팩토리얼 초기값 설정
    factorial = 1.0     # 타입 문제 때문에 실수형으로.
    
    # 반복을 통한 테일러 시리즈 계산..(무한하게는 할 수 없어서 반복 횟수 n을 지정해 근사하게 구한다)
    n = 100
    for i in range(1, n):
        A_power = np.dot(A_power, A)
        factorial *= i
        exp_A += A_power / factorial
    
    # 결과값
    print(exp_A)
    
    # (문제에서 최종적으로 알고자 하는 것) exp_A가 orthogonal 인지?
    # orthogonal matrix 정의로 검사
    identity_matrix = np.eye(A.shape[0])
    orthogonal_test = exp_A.T @ exp_A       # transpose와 inverse가 같아야 하니까
    
    print(orthogonal_test)
    
    if np.allclose(identity_matrix, orthogonal_test, atol=1e-10):       # 근사치니까 당연히 완벽히 identity가 나올 수는 없다
        print("when A is skew-symmetric, exponential A is orthogonal.")
    else:
        print("not orthogonal. you did something wrong...")
    
else:
    print(A)
    print("failed to generate skew-symmetric matrix")