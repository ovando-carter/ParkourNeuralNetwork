from datetime import datetime

# I should be able to modify this code so that the flux query finds the start and end time of the data
# Or I could even just input the values from the terminal
stringStart = '2022-03-01 22:21:20'

stringEnd = '2022-03-01 22:29:30'

start = datetime.strptime(stringStart, '%Y-0%m-0%d %H:%M:%S')

end = datetime.strptime(stringEnd, '%Y-0%m-0%d %H:%M:%S')

currentTime = datetime.now()


# we need to find the number of hours from the start time of the data to now, and from end time to now.
dataStart2Now = datetime.now() - start
dataEnd2Now = datetime.now() - end

print('start time: ', dataStart2Now)
print('end time: ',dataEnd2Now)


'''
# create a basic read out
inputStart_day = int(input('what is the number of days from start: '))
inputStart_hr = int(input('what is the number of hrs from start: '))
inputEnd_day = int(input('what is the number of days from End: '))
inputEnd_hr = int(input('what is the number of hrs from End: '))
'''



