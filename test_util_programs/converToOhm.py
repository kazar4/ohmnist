
# 30 is our failed case
band1 = [0,1,2,3,4,5,6,7,8,9,30,30]
band2 = [0,1,2,3,4,5,6,7,8,9,30,30]
band3 = [0,1,2,3,4,5,6,7,8,9,30,30]
band4 = [1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000,0.1,0.01]
band5 = [30, 1, 2, 30, 30, 0.5, 0.25, 0.10, 0.05, 30, 5, 10]

def returnOhms(data):
    if 30 in [band1[data[0]], band2[data[1]], band3[data[2]], band4[data[3]], band5[data[4]]]:
        print("Invalid resistor given")
        return None

    numbers = float(str(band1[data[0]]) + str(band2[data[1]]) + str(band3[data[2]]))
    numbers = numbers * float(band4[data[3]])
    text = f"{numbers} Ω ± {band5[data[4]]}%"
    return text

# invalid test case
print(returnOhms([10,3,7,5,8]))

# valid test case
print(returnOhms([3,3,7,5,8]))