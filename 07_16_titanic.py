import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

# [실습] titanic 승객 데이터 활용
# 컬럼 설명

# PassengerId - 승객 고유번호
# Survived - 생존유무(target 값) (0 = 사망, 1 = 생존)
# Pclass - 티켓 클래스 (1 = 1st, 2 = 2nd, 3 = 3rd)
# Name - 탑승객 성명
# Sex - 성별 (male: 남성, female: 여성)
# Age - 나이(세)
# SibSp - 함께 탑승한 형제자매, 배우자 수 총합
# Parch - 함께 탑승한 부모, 자녀 수 총합
# Ticket - 티켓 넘버
# Fare - 탑승 요금
# Cabin - 객실 넘버
# Embarked - 탑승 항구 (C: Cherbourg, Q: Queenstown, S: Southampton)
# # titanic.csv 파일 로드
# # titanic 데이터셋의 메타 데이터 확인

titanic = pd.read_csv('tasks/data/titanic.csv')
#print(titanic.values)
print(">----------------------------------------------------------<")

#-------------------------------------------------------------------------------

# 기초문제
#  1. 전체 승객 중 생존자의 수와 사망자의 수를 구하여 출력

total_survivor = titanic[titanic["Survived"] == 1]
total_death = titanic[titanic["Survived"] == 0]


print(total_survivor.value_counts("Survived"))
print(">----------------------------------------------------------<")
print(total_death.value_counts("Survived"))
print(">----------------------------------------------------------<")

#  2. 승객 나이의 평균 출력

total_age_sum = titanic["Age"].sum()
total_passenger = titanic["PassengerId"].max()
mean_age = total_age_sum / total_passenger
print(mean_age)
print(">----------------------------------------------------------<")

#  3. 1등급 승객 중 운임이 가장 높은 승객의 이름 출력

find_top_fare = titanic["Fare"].max()
find_top_fare_name = titanic[titanic["Fare"] == find_top_fare]["Name"]
print(find_top_fare_name)
print(">----------------------------------------------------------<")

#  4. 여성 승객의 수를 구하여 출력

find_female = titanic[titanic["Sex"] == "female"]
print(find_female.value_counts("Sex"))
print(">----------------------------------------------------------<")

#  5. 운임이 50 이상인 승객의 이름과 운임 출력

find_fare_over_50 = titanic[titanic["Fare"] >= 50]
find_name_fare_over_50 = find_fare_over_50[["Name", "Fare"]]
print(find_name_fare_over_50)
print(">----------------------------------------------------------<")

#  6. 나이가 18세 미만인 승객들의 생존율을 계산하여 출력

find_age_under_18 = titanic[titanic["Age"] < 18].count()
find_age_under_18_count = find_age_under_18["PassengerId"]
find_survivor_under_18 = titanic[titanic["Age"] < 18].value_counts("Survived")[1]
survival_rate = find_survivor_under_18 / find_age_under_18_count * 100 
print(survival_rate)
print(">----------------------------------------------------------<")

#  7. 2등급 객실에 탑승한 승객의 평균 운임을 계산하여 출력

find_pclass_2 = titanic[titanic["Pclass"] == 2]
find_pclass_2_fare = find_pclass_2["Fare"].sum()
find_pclass_2_counts = find_pclass_2.count()
find_pclass_2_count = find_pclass_2_counts["PassengerId"]
average_pclass_2 = find_pclass_2_fare / find_pclass_2_count
print(average_pclass_2)
print(">----------------------------------------------------------<")

#  8. 승객 중 나이가 가장 많은 사람의 나이와 이름을 출력

find_oldest_passenger = titanic["Age"].max()
find_oldest_name_age = titanic[titanic["Age"] == find_oldest_passenger][["Name", "Age"]]
print(find_oldest_name_age)
print(">----------------------------------------------------------<")

#  9. S 항구에서 탑승한 승객의 수를 출력

find_embarked_s_passenger_count = titanic[titanic["Embarked"] == "S"].count()["PassengerId"]
print(find_embarked_s_passenger_count)
print(">----------------------------------------------------------<")

#  10. 3등급 객실에 탑승한 남성 승객 중 생존자의 수를 출력

find_pclass_3_male_count = titanic[titanic["Pclass"] == 3].value_counts("Sex")["male"]
find_pclass_3_male_survivor = titanic[titanic["Pclass"] == 3].value_counts("Survived")[1]
pclass_3_survival_rate = find_pclass_3_male_survivor / find_pclass_3_male_count * 100
print(pclass_3_survival_rate)
print(">----------------------------------------------------------<")

# 고급문제
#  1. 생존자(Survived)와 그렇지 않은 사람들의 `Pclass`(객실 등급)별 평균 나이를 각각 출력

find_survivor = titanic[titanic["Survived"] == 1][["Pclass", "Age"]]
for pclass_num in range(1,4):
    find_survivor_placc_1_total_age = find_survivor.loc[find_survivor['Pclass'] == pclass_num, 'Age'].sum()
    find_survivor_placc_1_total_count = find_survivor.loc[find_survivor['Pclass'] == pclass_num, 'Age'].count()
    pclass_survivor_avg_age = find_survivor_placc_1_total_age / find_survivor_placc_1_total_count
    print(f"{pclass_num}등급 객실의 생존자 나이 평균 : {pclass_survivor_avg_age}")

find_death = titanic[titanic["Survived"] == 0][["Pclass", "Age"]]
for pclass_num in range(1,4):
    find_death_placc_1_total_age = find_death.loc[find_death['Pclass'] == pclass_num, 'Age'].sum()
    find_death_placc_1_total_count = find_death.loc[find_death['Pclass'] == pclass_num, 'Age'].count()
    pclass_death_avg_age = find_death_placc_1_total_age / find_death_placc_1_total_count
    print(f"{pclass_num}등급 객실의 사망자 나이 평균 : {pclass_death_avg_age}")
print(">----------------------------------------------------------<")

#  2. `SibSp`(형제/배우자 수)가 3 이상인 승객들 중에서, `Fare`(운임) 상위 5명의 이름과 나이를 출력

find_sibsp_over_3 = titanic[titanic["SibSp"] >= 3]
find_sibsp_top_5 = find_sibsp_over_3.sort_values(by="Age", ascending=False).head(5)[["Name", "Age"]]
print(find_sibsp_top_5)
print(">----------------------------------------------------------<")

#  3. 승객 중에서 `Age` 값이 결측치(NaN)인 사람들의 `Embarked`(탑승 항구)별 분포를 출력

find_age_nan_embarked = titanic[titanic["Age"].isna()]
embarked_counts = find_age_nan_embarked['Embarked'].value_counts()

for port in ['C', 'Q', 'S']:
    count = embarked_counts.get(port, 0)
    print(f"{port}: {count}개")

#  4. 나이가 16세 미만이고, 혼자 탑승한(`SibSp == 0` and `Parch == 0`) 승객의 이름과 성별, 생존 여부를 출력
#  5. 1등급(Pclass == 1) 객실에 탑승한 승객들 중에서, 운임(Fare)이 평균보다 높은 사람들의 이름과 나이, 운임을 출력
#  6. 나이(Age), 객실 등급(Pclass), 운임(Fare) 값에 결측치가 없는 승객들만 대상으로, 각 등급별 평균 운임과 중앙값 운임을 출력
#  7. 승객들의 `Embarked` 별 평균 나이를 계산하고, 가장 많은 승객이 탑승한 항구를 기준으로 생존율을 계산하여 출력
#  8. 승객 중에서 여성(`Sex == 'female'`)이면서 30세 이상인 사람들의 생존율을 계산하여 출력
#  9. 동반한 형제 또는 배우자 수(`SibSp`)와 부모 또는 자녀 수(`Parch`)의 합이 3 이상인 승객들만을 대상으로, 그들의 생존율을 계산하여 출력
#  10. `Cabin`(객실 번호) 정보가 있는 승객들만을 대상으로, 그들의 생존율을 객실 등급별로 구분하여 계산하여 출력